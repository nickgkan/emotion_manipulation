"""Train and test classification on ScanNet."""

import argparse
import os
import os.path as osp

import numpy as np
import pkbar
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from artemis_dataset import ArtEmisImageDataset
from early_stopping_scheduler import EarlyStopping
from metrics import compute_ap
from models.resnet_classifier import ResNetClassifier


def train_classifier(model, data_loaders, args):
    """Train a 3d object classifier."""
    # Setup
    device = args.device
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = EarlyStopping(
        optimizer, factor=0.3, mode='max', max_decays=1, patience=3
    )
    start_epoch = 0
    is_trained = False
    if osp.exists(args.classifier_ckpnt):
        checkpoint = torch.load(args.classifier_ckpnt)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = EarlyStopping(
            optimizer, factor=0.3, mode='max', max_decays=1, patience=3
        )
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        is_trained = checkpoint["is_trained"]
    if is_trained:
        return model
    writer = SummaryWriter('runs/' + args.checkpoint.replace('.pt', ''))

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print("Epoch: %d/%d" % (epoch + 1, args.epochs))
        kbar = pkbar.Kbar(target=len(data_loaders['train']), width=25)
        model.train()
        for step, ex in enumerate(data_loaders['train']):
            images, _, emotions = ex
            logits = model(images.to(device))
            labels = emotions.to(device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            kbar.update(step, [("loss", loss)])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar(
                'loss', loss.item(),
                epoch * len(data_loaders['train']) + step
            )
        writer.add_scalar(
            'lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch
        )
        # Evaluation and model storing
        print("\nValidation")
        acc = eval_classifier(model, data_loaders['test'], args)
        writer.add_scalar('mAP', acc, epoch)
        improved, ret_epoch, keep_training = scheduler.step(acc)
        if ret_epoch < epoch:
            scheduler.reduce_lr()
            checkpoint = torch.load(args.classifier_ckpnt)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Load checkpoint to update scheduler and epoch
        if osp.exists(args.classifier_ckpnt):
            checkpoint = torch.load(args.classifier_ckpnt)
        else:
            checkpoint = {"epoch": 0}
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        checkpoint["epoch"] += 1
        checkpoint["is_trained"] = not keep_training
        if improved:
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, args.classifier_ckpnt)
        if not keep_training:
            break
    # Test
    test_acc = eval_classifier(model, data_loaders['test'], args)
    print(f"Test Accuracy: {test_acc}")
    return model


@torch.no_grad()
def eval_classifier(model, data_loader, args):
    """Evaluate model on val/test data."""
    model.eval()
    device = args.device
    kbar = pkbar.Kbar(target=len(data_loader), width=25)
    gt = []
    pred = []
    for step, ex in enumerate(data_loader):
        images, _, emotions = ex
        pred.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
        gt.append(emotions.cpu().numpy())
        kbar.update(step)
    AP = compute_ap(np.concatenate(gt), np.concatenate(pred))

    print(f"\nAccuracy: {np.mean(AP)}")
    return np.mean(AP)


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
    args = argparser.parse_args()
    args.classifier_ckpnt = osp.join(args.checkpoint_path, args.checkpoint)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Data loaders for classification
    data_loaders = {
        mode: DataLoader(
            ArtEmisImageDataset(mode, args.im_path),
            batch_size=args.batch_size,
            shuffle=mode == 'train',
            drop_last=mode == 'train',
            num_workers=4
        )
        for mode in ('train', 'test')
    }

    # Train classifier
    model = ResNetClassifier(
        num_classes=len(data_loaders['train'].dataset.emotions),
        pretrained=True, freeze_backbone=True, layers=34
    )
    model = train_classifier(model.to(args.device), data_loaders, args)
    eval_classifier(model, data_loaders['test'], args)


if __name__ == "__main__":
    main()
