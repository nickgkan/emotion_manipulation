"""Main utilities for classification on ArtEmis."""

import cv2
import numpy as np
import pkbar
from pytorch_grad_cam import GradCAM
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from metrics import compute_ap
from train_test_utils import (
    load_from_ckpnt, unnormalize_imagenet_rgb,
    back2color
)


def train_classifier(model, data_loaders, args):
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
        #model.enable_grads()
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
            break
        writer.add_scalar(
            'lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch
        )
        # Evaluation and model storing
        if epoch % 2 == 0:
            print("\nValidation")
            acc = eval_classifier(model, data_loaders['test'], args, writer, epoch=epoch)
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
    test_acc = eval_classifier(model, data_loaders['test'], args, writer)
    print(f"Test Accuracy: {test_acc}")
    return model


def eval_classifier(model, data_loader, args, writer=None, epoch=0):
    """Evaluate model on val/test data."""
    model.eval()
    #model.enable_all_grads()
    device = args.device
    kbar = pkbar.Kbar(target=len(data_loader), width=25)
    gt = []
    pred = []
    cam = GradCAM(
        model=model, target_layer=model.net.layer4[-1],
        use_cuda=True if torch.cuda.is_available() else False
    )
    for step, ex in enumerate(data_loader):
        images, _, emotions, _ = ex
        images = images.to(device)
        pred.append(torch.sigmoid(model(images)).detach().cpu().numpy())
        gt.append(emotions.cpu().numpy())
        kbar.update(step)
        # Log
        writer.add_image(
            'image_sample',
            back2color(unnormalize_imagenet_rgb(images[1], device)),
            epoch * len(data_loader) + step
        )
        for emo_id in torch.nonzero(emotions[1]).reshape(-1):
            grayscale_cam = cam(
                input_tensor=images[1:2],
                target_category=emo_id.item()
            )
            #grayscale_cam = grayscale_cam[0]
            '''
            writer.add_image(
                'gray_grad_cam_{}'.format(emo_id.item()),
                torch.from_numpy(np.uint8(255*grayscale_cam)).unsqueeze(0).repeat(3,1,1),
                epoch * len(data_loader) + step
            )
            '''
            heatmap = cv2.cvtColor(
                cv2.applyColorMap(np.uint8(255*grayscale_cam), cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB
            )
            heatmap = torch.from_numpy(np.float32(heatmap) / 255).to(device)
            rgb_img = unnormalize_imagenet_rgb(images[1], device)
            rgb_cam_vis = heatmap.permute(2, 0, 1).contiguous() + rgb_img
            rgb_cam_vis = rgb_cam_vis / torch.max(rgb_cam_vis).item()
            writer.add_image(
                'image_grad_cam_{}'.format(emo_id.item()),
                back2color(rgb_cam_vis),
                epoch * len(data_loader) + step
            )
    AP = compute_ap(np.concatenate(gt), np.concatenate(pred))

    print(f"\nAccuracy: {np.mean(AP)}")
    print(AP)
    #model.zero_grad()
    #model.disable_all_grads()
    return np.mean(AP)
