"""Main utilities for classification on ArtEmis."""

import cv2
import numpy as np
import pkbar
from pytorch_grad_cam import GradCAM
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.
from tensorboardX import SummaryWriter

from train_test_utils import (
    load_from_ckpnt, unnormalize_imagenet_rgb,
    back2color
)


def train_bin_classifier(model, data_loaders, args):
    """Train an emotion classifier."""
    # Setup
    device = args.device
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    model, optimizer, _, start_epoch, is_trained = load_from_ckpnt(
        args.classifier_ckpnt, model, optimizer
    )
    scheduler = MultiStepLR(optimizer, [3, 6, 9], gamma=0.3,
                            last_epoch=start_epoch - 1)
    if is_trained:
        return model
    writer = SummaryWriter('runs/' + args.checkpoint.replace('.pt', ''))
    best_acc = -1

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print("Epoch: %d/%d" % (epoch + 1, args.epochs))
        kbar = pkbar.Kbar(target=len(data_loaders['train']), width=25)
        model.train()
        for step, ex in enumerate(data_loaders['train']):
            images, _, _, neg_images = ex
            labels = torch.cat((
                torch.ones(len(images)), torch.zeros(len(images))
            )).to(device)
            images = torch.cat((images, neg_images))
            logits = model(images.to(device)).squeeze(1)
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
        acc = eval_bin_classifier(model, data_loaders['test'], args, writer)
        writer.add_scalar('mAP', acc, epoch)
        if acc >= best_acc:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                args.classifier_ckpnt
            )
            best_acc = acc
        else:  # load checkpoint to update epoch
            checkpoint = torch.load(args.classifier_ckpnt)
            checkpoint["epoch"] += 1
            torch.save(checkpoint, args.classifier_ckpnt)
        scheduler.step()
    # Test
    test_acc = eval_bin_classifier(model, data_loaders['test'], args)
    print(f"Test Accuracy: {test_acc}")
    return model


def eval_bin_classifier(model, data_loader, args, writer=None):
    """Evaluate model on val/test data."""
    model.eval()
    device = args.device
    kbar = pkbar.Kbar(target=len(data_loader), width=25)
    gt = 0
    pred = 0
    '''
    cam = GradCAM(
        model=model, target_layer=model.net.layer4[-1],
        use_cuda=True if torch.cuda.is_available() else False
    )
    '''
    for step, ex in enumerate(data_loader):
        images, _, _, neg_images = ex
        labels = torch.cat((
            torch.ones(len(images)), torch.zeros(len(neg_images))
        )).numpy()
        images = torch.cat((images, neg_images)).to(device)
        with torch.no_grad():
            pred += (
                (torch.sigmoid(model(images)).squeeze(-1).cpu().numpy() > 0.5) * 1
                == labels
            ).sum().item()
        gt += len(images)
        kbar.update(step)
        if step > 0:
            continue
        # Log
        '''
        writer.add_image(
            'image_sample',
            back2color(unnormalize_imagenet_rgb(images[0], device)),
            step
        )
        grayscale_cam = cam(
            input_tensor=images[0:1],
            target_category=0
        )
        grayscale_cam = grayscale_cam[0]
        heatmap = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255*grayscale_cam), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB
        )
        heatmap = torch.from_numpy(np.float32(heatmap) / 255).to(device)
        rgb_img = unnormalize_imagenet_rgb(images[0], device)
        rgb_cam_vis = heatmap.permute(2, 0, 1).contiguous() + rgb_img
        rgb_cam_vis = rgb_cam_vis / torch.max(rgb_cam_vis).item()
        writer.add_image(
            'image_grad_cam',
            back2color(rgb_cam_vis),
            step
        )
        '''

    print(f"\nAccuracy: {pred / gt}")
    return pred / gt
