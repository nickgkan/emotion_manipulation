"""Dataset utilities for ArtEmis."""

import csv
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ArtEmisDataset(Dataset):
    """
    General dataset class for ArtEmis.

    Inputs:
        split (str): train/val/test
        im_path (str): path to images
        seed (int): random seed for dataset splitting
        emot_label (str): emotion label to filter images
        im_size (int): resized image dimension
    """

    def __init__(self, split, im_path, seed=1184, emot_label=None, im_size=64):
        """Initialize task-independent dataset."""
        super().__init__()
        self.split = split
        self.im_path = im_path
        self.emot_label = emot_label
        self.im_size = im_size
        self.annos, self.neg_annos = self.load_annotations(split, seed)
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
        if self.emot_label is None:
            return [annos[img] for img in imgs], None
        return (
            [
                annos[img] for img in imgs
                if self.emot_label in annos[img]['emotion']
            ],
            [
                annos[img] for img in imgs
                if self.emot_label not in annos[img]['emotion']
            ]
        )

    @staticmethod
    def _to_img_wise(annos):
        """Convert annotation list to image-wise annotations."""
        per_img = dict()
        for anno in annos:
            if anno['painting'] not in per_img:
                per_img[anno['painting']] = {
                    'art_style': anno['art_style'],
                    'emotion': [],
                    'utterance': set(),
                    'painting': anno['painting'] + '_resize'
                }
            per_img[anno['painting']]['emotion'].append(anno['emotion'])
            per_img[anno['painting']]['utterance'].add(anno['utterance'])
        return per_img

    def _load_image(self, img_name):
        """Load image and add augmentations."""
        img_name = os.path.join(self.im_path, img_name)
        _img = Image.open(img_name)
        width, height = _img.size
        max_wh = max(width, height)
        mean_ = [0.485, 0.456, 0.406]
        std_ = [0.229, 0.224, 0.225]
        size = self.im_size
        if self.split == 'train':
            preprocessing = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(20),
                transforms.RandomPerspective(0.1, 0.5),
                transforms.Pad((0, 0, max_wh - width, max_wh - height)),
                transforms.Resize((size + 8, size + 8)),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean_, std_),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
            ])
            preprocessing = transforms.Compose([
                transforms.Pad((0, 0, max_wh - width, max_wh - height)),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean_, std_)
            ])
        else:
            preprocessing = transforms.Compose([
                transforms.Pad((0, 0, max_wh - width, max_wh - height)),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean_, std_)
            ])
        return preprocessing(_img)  # (1, H, W, 3)

    def __len__(self):
        """Return number of annotations."""
        return len(self.annos)

    def __getitem__(self, index):
        """Return a sample to form batch."""
        index = 2
        anno = self.annos[index]
        # Load image
        img = self._load_image("{0}/{1}.jpg".format(
            anno['art_style'], anno['painting']
        ))
        # Art-style to index
        style = self.styles[anno['art_style']]
        # Emotions to index
        emotions = np.zeros((len(self.emotions),))
        emotions[list(map(self.emotions.get, anno['emotion']))] = 1
        # Bring a negative img if emotion is specified
        neg_img = [1]
        if self.neg_annos is not None:
            neg = self.neg_annos[np.random.randint(0, len(self.neg_annos) - 1)]
            neg_img = self._load_image("{0}/{1}.jpg".format(
                neg['art_style'], neg['painting']
            ))
        return img, style, emotions, neg_img
