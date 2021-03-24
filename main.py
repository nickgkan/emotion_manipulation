"""Train and test classification on ScanNet."""

import argparse
import os
import os.path as osp

import pkbar
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from artemis_dataset import ArtEmisImageDataset
# from src.tools.early_stopping_scheduler import EarlyStopping
from early_stopping_scheduler import EarlyStopping

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
        # Evaluation and model storing
        print("\nValidation")
        acc = eval_classifier(model, data_loaders['test'], args)
        improved, ret_epoch, keep_training = scheduler.step(acc)
        if ret_epoch < epoch:
            scheduler.reduce_lr()
            checkpoint = torch.load(args.classifier_ckpnt)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Load checkpoint to update scheduler and epoch
        checkpoint = torch.load(args.classifier_ckpnt)
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
    num_correct = 0
    num_examples = 0
    for step, ex in enumerate(data_loader):
        images, _, emotions = ex
        logits = model(images.to(device))
        # this is not a correct metric, re-implement better
        logits = logits.argmax(1).cpu()
        num_correct += (logits == emotions.argmax(1)).sum()
        num_examples += len(logits)
        kbar.update(step, [("accuracy", num_correct / num_examples)])

    print(f"\nAccuracy: {num_correct / num_examples}")
    return num_correct / num_examples


def main():
    """Run main training/test pipeline."""
    data_path = "/projects/katefgroup/language_grounding/"
    data_path = "/projects/katefgroup/viewpredseg/art/"
    if not osp.exists(data_path):
        data_path = 'data/'  # or change this if you work locally

    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--im_path", default=osp.join(data_path, "images/")
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
    os.makedirs(args.checkpoint, exist_ok=True)

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
    model = ResNetClassifier(num_classes=8, pretrained=True, layers=50)
    model = train_classifier(model.to(args.device), data_loaders, args)
    eval_classifier(model, data_loaders['test'], args)


if __name__ == "__main__":
    main()
