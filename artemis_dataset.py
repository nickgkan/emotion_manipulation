"""Dataset utilities for ArtEmis."""

import csv
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import ipdb
st = ipdb.set_trace


class ArtEmisDataset(Dataset):
    """
    General dataset class for ArtEmis.

    Inputs:
        split (str): train/val/test
        im_path (str): path to images
        seed (int): random seed for dataset splitting
    """

    def __init__(self, split, im_path, seed=1184):
        """Initialize task-independent dataset."""
        super().__init__()
        self.split = split
        self.im_path = im_path
        self.annos = self.load_annotations(split, seed)
        self.styles, self.emotions = self._get_classes()

    @staticmethod
    def _read_from_csv():
        with open('artemis_dataset_release_v0.csv') as fid:
            csv_reader = csv.reader(fid)
            names2id = {name: n for n, name in enumerate(next(csv_reader))}
            annos = [
                {
                    'art_style': line[names2id['art_style']],
                    'painting': str.encode(
                        line[names2id['painting']], 'ascii', errors="ignore"
                    ).decode("utf-8"),
                    'emotion': line[names2id['emotion']],
                    'utterance': line[names2id['utterance']]
                }
                for line in csv_reader
            ]
        return annos

    def _get_classes(self):
        annos = self._read_from_csv()
        styles = sorted(list(set(anno['art_style'] for anno in annos)))
        emotions = sorted(list(set(anno['emotion'] for anno in annos)))
        return (
            {style: s for s, style in enumerate(styles)},
            {emotion: e for e, emotion in enumerate(emotions)}
        )

    def _sample_split_indices(self, len_inds, split, seed):
        np.random.seed(seed)
        inds = np.random.permutation(np.arange(len_inds))
        if split == 'train':
            inds = inds[:int(0.8 * len(inds))]
        elif split == 'val':
            inds = inds[int(0.8 * len(inds)):int(0.9 * len(inds))]
        else:
            inds = inds[int(0.9 * len(inds)):]
        return inds

    @staticmethod
    def load_annotations(split, seed):
        """Load annotations (abstract method)."""
        return []

    def _load_image(self, img_name):
        """Load image and add augmentations."""
        img_name = os.path.join(self.im_path, img_name)
        mean_ = [0.485, 0.456, 0.406]
        std_ = [0.229, 0.224, 0.225]
        size = 224
        if self.split == 'train':
            preprocessing = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(20),
                transforms.RandomPerspective(0.1, 0.5),
                transforms.Resize((size + 16, size + 16)),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean_, std_),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
            ])
        else:
            preprocessing = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean_, std_)
            ])
        return preprocessing(Image.open(img_name))  # (1, H, W, 3)

    def __len__(self):
        """Return number of annotations."""
        return len(self.annos)

    def __getitem__(self, index):
        """Return a sample to form batch."""
        return []  # re-implement in child!


class ArtEmisImageDataset(ArtEmisDataset):
    """Image-based dataset for ArtEmis."""

    def __init__(self, split, im_path, seed=1184):
        """Initialize dataset."""
        super().__init__(split, im_path, seed)

    def load_annotations(self, split, seed):
        """Load annotations."""
        # Read from csv
        annos = self._read_from_csv()
        # Adapt to task
        annos = self._to_img_wise(annos)
        # Split
        imgs = sorted(list(annos.keys()))
        # imgs = sorted(list(set(anno['painting'] for anno in annos)))
        inds = self._sample_split_indices(len(imgs), split, seed)
        imgs = np.asarray(imgs)[inds].tolist()
        return [annos[img] for img in imgs]

    @staticmethod
    def _to_img_wise(annos):
        """Convert annotation list to image-wise annotations."""
        per_img = dict()
        for anno in annos:
            if anno['painting'] not in per_img:
                per_img[anno['painting']] = {
                    'painting': anno['painting'],
                    'art_style': anno['art_style'],
                    'orig': anno['orig'],
                    'new': anno['new'],
                    'emotion': set(),
                    'utterance': set()
                }
            per_img[anno['painting']]['emotion'].add(anno['emotion'])
            per_img[anno['painting']]['utterance'].add(anno['utterance'])
        return per_img

    def __getitem__(self, index):
        """Return a sample to form batch."""
        anno = self.annos[index]
        # Load image
        img = self._load_image("{0}/{1}.jpg".format(
            anno['art_style'], anno['painting'])
        )
        # Art-style to index
        style = self.styles[anno['art_style']]
        # Emotions to index
        emotions = np.zeros((len(self.emotions),))
        for emotion in anno['emotion']:
            emotions[self.emotions[emotion]] = 1
        return img, style, emotions
