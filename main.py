"""Train and test classification on ArtEmis."""

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

from artemis_dataset import ArtEmisDataset
from early_stopping_scheduler import EarlyStopping
from metrics import compute_ap
from models.resnet_classifier import ResNetClassifier, requires_grad
from models.resnet_ebm import ResNetEBM
from pytorch_grad_cam import GradCAM
import cv2

import ipdb
st = ipdb.set_trace


def unnormalize_imagenet_rgb(image, device):
    """Unnormalize normalized rgb using imagenet stats."""
    mean_ = torch.as_tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(device)
    std_ = torch.as_tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(device)
    image = (image * std_) + mean_
    return image


def back2color(image):
    return (image*255).type(torch.ByteTensor)


def load_from_ckpnt(ckpnt, model, optimizer=None, scheduler=None):
    """Load trained parameters from given checkpoint."""
    start_epoch = 0
    is_trained = False
    if osp.exists(ckpnt):
        checkpoint = torch.load(ckpnt)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        is_trained = checkpoint.get("is_trained", False)
    return model, optimizer, scheduler, start_epoch, is_trained


def langevin_updates(model, neg_samples, nsteps, langevin_lr):
    """Apply nsteps iterations of Langevin dynamics on neg_samples."""
    # Deactivate model gradients
    requires_grad(model.parameters(), False)
    model.eval()
    # Activate samples gradients
    neg_samples.requires_grad = True
    noise = torch.randn_like(neg_samples).to(neg_samples.device)  # noise
    # Langevin steps
    negs = [torch.clone(neg_samples[0]).detach()]  # for visualization
    for k in range(nsteps):
        # Noise
        noise.normal_(0, 0.005)
        neg_samples.data.add_(noise.data)
        # Forward-backward
        neg_out = model(neg_samples)
        neg_out.sum().backward()
        # Update neg_samples
        neg_samples.grad.data.clamp_(-0.01, 0.01)
        neg_samples.data.add_(-langevin_lr, neg_samples.grad.data)
        # Zero gradients
        neg_samples.grad.detach_()
        neg_samples.grad.zero_()
        # Clamp
        neg_samples.data.clamp(-0.485 / 0.224, (1 - 0.406) / 0.224)
        # Store intermediate results for visualization
        negs.append(torch.clone(neg_samples[0]).detach())
    # Detach samples
    neg_samples = neg_samples.detach()
    # Reactivate model gradients
    requires_grad(list(model.fc.parameters()), True)
    model.train()
    return neg_samples, negs


@torch.no_grad()
def clip_grad(parameters, optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]

            if 'step' not in state or state['step'] < 1:
                continue

            step = state['step']
            exp_avg_sq = state['exp_avg_sq']
            _, beta2 = group['betas']

            bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
            p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))


def train_classifier(model, data_loaders, args):
    """Train an emotion classifier."""
    # Setup
    device = args.device
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = EarlyStopping(
        optimizer, factor=0.3, mode='max', max_decays=1, patience=3
    )
    model, optimizer, scheduler, start_epoch, is_trained = load_from_ckpnt(
        args.classifier_ckpnt, model, optimizer, scheduler
    )
    if is_trained:
        return model
    writer = SummaryWriter('runs/' + args.checkpoint.replace('.pt', ''))

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print("Epoch: %d/%d" % (epoch + 1, args.epochs))
        kbar = pkbar.Kbar(target=len(data_loaders['train']), width=25)
        model.train()
        for step, ex in enumerate(data_loaders['train']):
            images, _, emotions, _ = ex
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


def train_generator(model, data_loaders, args):
    """Train an emotion EBM."""
    device = args.device
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    model, optimizer, _, start_epoch, is_trained = load_from_ckpnt(
        args.classifier_ckpnt, model, optimizer, scheduler=None
    )
    if is_trained:
        return model
    writer = SummaryWriter('runs/' + args.checkpoint.replace('.pt', ''))

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print("Epoch: %d/%d" % (epoch + 1, args.epochs))
        kbar = pkbar.Kbar(target=len(data_loaders['train']), width=25)
        model.train()
        for step, ex in enumerate(data_loaders['train']):
            images, _, emotions, neg_images = ex
            # positive samples
            pos_samples = images.to(device)
            # negative samples
            neg_samples, neg_list = langevin_updates(
                model, torch.clone(neg_images.to(device)),
                args.langevin_steps, args.langevin_step_size
            )
            neg_img_samples, _ = langevin_updates(
                model, torch.randn_like(pos_samples).to(device),
                args.langevin_steps, args.langevin_step_size
            )
            # Compute energy
            pos_out = model(pos_samples)
            neg_out = model(neg_samples)
            neg_img_out = model(neg_images.to(device))
            neg_img_ld_out = model(neg_img_samples)
            # Loss
            loss = (
                pos_out**2 + neg_out**2 + neg_img_out**2 + neg_img_ld_out**2
                + 3*pos_out - neg_out - neg_img_out - neg_img_ld_out
            ).mean()
            # Step
            optimizer.zero_grad()
            loss.backward()
            clip_grad(model.parameters(), optimizer)
            optimizer.step()
            kbar.update(step, [("loss", loss)])
            # Log loss
            writer.add_scalar(
                'loss', loss.item(),
                epoch * len(data_loaders['train']) + step
            )
            # Log image evolution
            writer.add_image(
                'random_image_sample',
                back2color(unnormalize_imagenet_rgb(pos_samples[0], device)),
                epoch * len(data_loaders['train']) + step
            )
            vid_to_write = torch.stack(neg_list, dim=0).unsqueeze(0)
            writer.add_video(
                'ebm_evolution', vid_to_write, fps=args.ebm_log_fps,
                global_step=epoch * len(data_loaders['train']) + step
            )
        writer.add_scalar(
            'lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch
        )
        # Save checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            },
            args.classifier_ckpnt
        )
        print('\nValidation')
        print(eval_generator(model, data_loaders['test'], args))
    return model


@torch.no_grad()
def eval_classifier(model, data_loader, args):
    """Evaluate model on val/test data."""
    model.eval()
    device = args.device
    kbar = pkbar.Kbar(target=len(data_loader), width=25)
    gt = []
    pred = []
    cam = GradCAM(model=model, target_layer=model.layer4[-1], use_cuda=True if torch.cuda.is_available() else False)
    for step, ex in enumerate(data_loader):
        images, _, emotions, _ = ex
        pred.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
        gt.append(emotions.cpu().numpy())
        kbar.update(step)
        # Log
        writer.add_image(
            'image_sample', back2color(unnormalize_imagenet_rgb(images[0], device)),
            step
        )
        for emo_id in torch.nonzero(emotions[0]).reshape(-1):
            grayscale_cam = cam(input_tensor=images[0:1], target_category=emo_id.item())
            heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*grayscale_cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heatmap = torch.from_numpy(np.float32(heatmap) / 255).to(device)
            rgb_cam_vis = heatmap + unnormalize_imagenet_rgb(images[0], device)
            rgb_cam_vis = rgb_cam_vis / torch.max(rgb_cam_vis)
            writer.add_image(
                'image_grad_cam_{}'.format(emo_id.item()), back2color(rgb_cam_vis),
                step
            )
    AP = compute_ap(np.concatenate(gt), np.concatenate(pred))

    print(f"\nAccuracy: {np.mean(AP)}")
    return np.mean(AP)


@torch.no_grad()
def eval_generator(model, data_loader, args):
    """Evaluate model on val/test data."""
    model.eval()
    device = args.device
    kbar = pkbar.Kbar(target=len(data_loader), width=25)
    gt = 0
    pred = 0
    for step, ex in enumerate(data_loader):
        images, _, _, neg_images = ex
        # Compute energy
        pos_out = model(images.to(device))
        neg_img_out = model(neg_images.to(device))
        gt += len(images)
        pred += (pos_out < neg_img_out).sum()
        kbar.update(step, [("acc", pred / gt)])

    print(f"\nAccuracy: {pred / gt}")
    return pred / gt


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
    argparser.add_argument("--run_classifier", action='store_true')
    argparser.add_argument("--run_generator", action='store_true')
    argparser.add_argument("--emot_label", default=None)
    args = argparser.parse_args()
    args.classifier_ckpnt = osp.join(args.checkpoint_path, args.checkpoint)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Data loaders for classification
    data_loaders = {
        mode: DataLoader(
            ArtEmisDataset(mode, args.im_path, emot_label=args.emot_label),
            batch_size=args.batch_size,
            shuffle=mode == 'train',
            drop_last=mode == 'train',
            num_workers=4
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
        eval_classifier(model.to(args.device), data_loaders['test'], args)

    # Train generator
    if args.run_generator:
        model = ResNetEBM(
            pretrained=True, freeze_backbone=True, layers=50
        )
        model = train_generator(model.to(args.device), data_loaders, args)
        eval_generator(model.to(args.device), data_loaders['test'], args)


if __name__ == "__main__":
    main()
