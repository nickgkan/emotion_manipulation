"""Train and test classification on ArtEmis."""

import argparse
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from artemis_dataset import ArtEmisDataset
from models.resnet_classifier import ResNetClassifier
from models.resnet_ebm import ResNetEBM
from train_bin_classifier import train_bin_classifier, eval_bin_classifier
from train_classifier import train_classifier, eval_classifier
from train_generator import train_generator, eval_generator
from train_manipulator import train_manipulator, eval_manipulator
from train_transformations import train_transformations, eval_transformations
from torch.utils.tensorboard import SummaryWriter

import ipdb
st = ipdb.set_trace


def main():
    """Run main training/test pipeline."""
    data_path = "/projects/katefgroup/viewpredseg/art/"
    if not osp.exists(data_path):
        data_path = 'data/'  # or change this if you work locally

    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--im_path", default=osp.join(data_path, "wikiart/")
    )
    argparser.add_argument(
        "--checkpoint_path", default=osp.join(data_path, "checkpoints/")
    )
    argparser.add_argument("--checkpoint", default="classifier.pt")
    argparser.add_argument("--epochs", default=50, type=int)
    argparser.add_argument("--batch_size", default=128, type=int)
    argparser.add_argument("--lr", default=1e-3, type=float)
    argparser.add_argument("--wd", default=1e-5, type=float)
    argparser.add_argument("--langevin_steps", default=20, type=int)
    argparser.add_argument("--langevin_step_size", default=10, type=float)
    argparser.add_argument("--ebm_log_fps", default=6, type=int)
    argparser.add_argument("--run_bin_classifier", action='store_true')
    argparser.add_argument("--run_classifier", action='store_true')
    argparser.add_argument("--run_generator", action='store_true')
    argparser.add_argument("--run_manipulator", action='store_true')
    argparser.add_argument("--run_transformations", action='store_true')
    argparser.add_argument("--emot_label", default=None)
    args = argparser.parse_args()
    args.classifier_ckpnt = osp.join(args.checkpoint_path, args.checkpoint)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Data loaders for classification
    data_loaders = {
        mode: DataLoader(
            ArtEmisDataset(
                mode, args.im_path, emot_label=args.emot_label,
                im_size=224 if args.run_classifier else 64
            ),
            batch_size=args.batch_size,
            shuffle=mode == 'train',
            drop_last=mode == 'train',
            num_workers=0
        )
        for mode in ('train', 'test')
    }

    # Train classifier
    # Emotion labels
    # {'amusement': 0, 'anger': 1, 'awe': 2, 'contentment': 3, 'disgust': 4,
    # 'excitement': 5, 'fear': 6, 'sadness': 7, 'something else': 8}
    if args.run_classifier:
        model = ResNetClassifier(
            num_classes=len(data_loaders['train'].dataset.emotions),
            pretrained=True, freeze_backbone=True, layers=34
        )
        model = train_classifier(model.to(args.device), data_loaders, args)
        eval_writer = SummaryWriter('runs/classifier_eval')
        eval_classifier(model, data_loaders['test'], args, eval_writer)

    # Train binary classifier
    if args.run_bin_classifier:
        model = ResNetClassifier(
            num_classes=1,
            pretrained=True, freeze_backbone=True, layers=18
        )
        model = train_bin_classifier(model.to(args.device), data_loaders, args)
        eval_writer = SummaryWriter('runs/bin_classifier_eval')
        eval_bin_classifier(model, data_loaders['test'], args, eval_writer)

    # Train generator
    if args.run_generator:
        model = ResNetEBM(
            pretrained=False, freeze_backbone=False, layers=18
        )
        model = train_generator(model.to(args.device), data_loaders, args)
        eval_generator(model.to(args.device), data_loaders['test'], args)

    # Train manipulator
    if args.run_manipulator:
        model = ResNetEBM(
            pretrained=False, freeze_backbone=False, layers=34
        )
        model = train_manipulator(model.to(args.device), data_loaders, args)
        eval_manipulator(model.to(args.device), data_loaders['test'], args)

    # Train transformations
    if args.run_transformations:
        model = ResNetEBM(
            pretrained=False, freeze_backbone=False, layers=18
        )
        model = train_transformations(model.to(args.device), data_loaders, args)
        eval_transformations(model.to(args.device), data_loaders['test'], args)


if __name__ == "__main__":
    main()
