"""Train Transformations"""

import pkbar
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from train_test_utils import (
    load_from_ckpnt, clip_grad, back2color, unnormalize_imagenet_rgb, normalize_imagenet_rgb,
    random_brightness, random_contrast, random_saturation, random_linear, rand_augment
)

import ipdb
st = ipdb.set_trace


def train_transformations(model, data_loaders, args):
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
        model.disable_batchnorm()
        model.zero_grad()
        # model.enable_grads()
        for step, ex in enumerate(data_loaders['train']):
            images, _, emotions, neg_images = ex
            # positive samples
            pos_samples = images.to(device)
            # prepare negative samples
            neg_samples = rand_augment(images.clone().to(device))
            # negative samples
            neg_ld_samples, neg_list = langevin_updates(
                model, torch.clone(neg_samples),
                args.langevin_steps, args.langevin_step_size,
            )
            # Compute energy
            pos_out = model(normalize_imagenet_rgb(pos_samples))
            neg_img_out = model(normalize_imagenet_rgb(neg_images.to(device)))
            neg_ld_out = model(normalize_imagenet_rgb(neg_ld_samples.to(device)))
            # Loss
            loss_reg = (pos_out**2 + neg_ld_out**2 + neg_img_out**2).mean()
            # loss_reg = (torch.abs(pos_out) + torch.abs(neg_ld_out) + torch.abs(neg_img_out)).mean()
            loss_ml = 2*pos_out.mean() - neg_ld_out.mean() - neg_img_out.mean()
            coeff = loss_ml.detach().clone() / loss_reg.detach().clone()
            loss = 0.5*loss_reg + loss_ml
            # if epoch == 0:
            #     loss = loss * 0.05
            '''
            loss = (
                pos_out**2 + neg_out**2 + neg_img_out**2 + neg_img_ld_out**2
                + 3*pos_out - neg_out - neg_img_out - neg_img_ld_out
            ).mean()
             '''
            # Step
            optimizer.zero_grad()
            loss.backward()
            clip_grad(model.parameters(), optimizer)
            optimizer.step()
            kbar.update(step, [("loss", loss)])
            # Log loss
            writer.add_scalar('energy/energy_pos', pos_out.mean().item(), epoch * len(data_loaders['train']) + step)
            writer.add_scalar('energy/energy_neg', neg_ld_out.mean().item(), epoch * len(data_loaders['train']) + step)
            writer.add_scalar('loss/loss_reg', loss_reg.item(), epoch * len(data_loaders['train']) + step)
            writer.add_scalar('loss/loss_ml', loss_ml.item(), epoch * len(data_loaders['train']) + step)
            writer.add_scalar('loss/loss_total', loss.item(), epoch * len(data_loaders['train']) + step)
            # Log image evolution
            if step % 50 != 0:
                continue
            writer.add_image(
                'ld/random_image_sample',
                back2color(pos_samples[0]),
                epoch * len(data_loaders['train']) + step
            )
            writer.add_image(
                'ld/ld_start',
                back2color(neg_list[0]),
                epoch * len(data_loaders['train']) + step
            )
            writer.add_image(
                'ld/ld_end',
                back2color(neg_list[-1]),
                epoch * len(data_loaders['train']) + step
            )
            neg_list = [
                back2color(neg)
                for neg in neg_list
            ]
            neg_list = [torch.zeros_like(neg_list[0])] + neg_list
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
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            },
            "transformations_%02d.pt" % (epoch+1)
        )
        print('\nValidation')
        print(eval_transformations(model, data_loaders['test'], args))
    return model


@torch.no_grad()
def eval_transformations(model, data_loader, args):
    """Evaluate model on val/test data."""
    model.eval()
    model.disable_batchnorm()
    device = args.device
    kbar = pkbar.Kbar(target=len(data_loader), width=25)
    gt = 0
    pred = 0
    for step, ex in enumerate(data_loader):
        images, _, _, neg_images = ex
        # Compute energy
        pos_out = model(normalize_imagenet_rgb(images.to(device)))
        # negative samples
        neg_samples = rand_augment(images.clone().to(device))
        neg_img_out = model(normalize_imagenet_rgb(neg_samples.to(device)))
        gt += len(images)
        pred += (pos_out < neg_img_out).sum()
        kbar.update(step, [("acc", pred / gt)])

    print(f"\nAccuracy: {pred / gt}")
    return pred / gt


def langevin_updates(model, neg_samples, nsteps, langevin_lr):
    """Apply nsteps iterations of Langevin dynamics on neg_samples."""
    # Deactivate model gradients
    model.disable_all_grads()
    model.eval()
    model.disable_batchnorm()
    # Image gradients are not needed here, as images are fixed
    neg_samples.requires_grad = False
    noise = torch.randn_like(neg_samples).to(neg_samples.device)  # noise
    # Transformation params: B * [1 (brightness) + 1 (contrast) + 1 (saturation) + 1 (color linear)]
    B, _, _, _ = neg_samples.shape
    brightness_params = torch.ones((B)).float().to(neg_samples.device)
    brightness_params.requires_grad = True
    contrast_params = torch.ones((B)).float().to(neg_samples.device)
    contrast_params.requires_grad = True
    saturation_params = torch.ones((B)).float().to(neg_samples.device)
    saturation_params.requires_grad = True
    linear_params_w = torch.eye(3).unsqueeze(0).repeat(B,1,1).to(neg_samples.device)
    linear_params_w.requires_grad = True
    linear_params_b = torch.zeros((B, 3)).float().to(neg_samples.device)
    linear_params_b.requires_grad = True
    # Langevin steps
    negs = [torch.clone(neg_samples[0]).detach()]  # for visualization
    for k in range(nsteps):
        # Noise
        #noise.normal_(0, 0.003)
        #neg_samples.data.add_(noise.data)
        # transformations
        trans_samples = neg_samples.clone()
        trans_samples = random_linear(trans_samples, linear_params_w, linear_params_b)
        trans_samples = random_brightness(trans_samples, brightness_params)
        trans_samples = random_contrast(trans_samples, contrast_params)
        trans_samples = random_saturation(trans_samples, saturation_params)
        # Clamp
        trans_samples.data.clamp_(-2.5, 2.5)
        # Forward-backward
        trans_out = model(normalize_imagenet_rgb(trans_samples))
        trans_out.sum().backward()
        # Update transform params
        brightness_params.grad.data.clamp_(-0.01, 0.01)
        brightness_params.data.add_(brightness_params.grad.data, alpha=-langevin_lr)
        contrast_params.grad.data.clamp_(-0.01, 0.01)
        contrast_params.data.add_(contrast_params.grad.data, alpha=-langevin_lr)
        saturation_params.grad.data.clamp_(-0.01, 0.01)
        saturation_params.data.add_(saturation_params.grad.data, alpha=-langevin_lr)
        #linear_params_w.grad.data.clamp_(-0.01, 0.01)
        #linear_params_w.data.add_(linear_params_w.grad.data, alpha=-langevin_lr)
        #linear_params_b.grad.data.clamp_(-0.01, 0.01)
        #linear_params_b.data.add_(linear_params_b.grad.data, alpha=-langevin_lr)
        # Zero gradients
        brightness_params.grad.detach_()
        brightness_params.grad.zero_()
        contrast_params.grad.detach_()
        contrast_params.grad.zero_()
        saturation_params.grad.detach_()
        saturation_params.grad.zero_()
        #linear_params_w.grad.detach_()
        #linear_params_w.grad.zero_()
        #linear_params_b.grad.detach_()
        #linear_params_b.grad.zero_()
        # Store intermediate results for visualization
        negs.append(torch.clone(trans_samples[0]).detach())
    # Detach samples
    trans_samples = trans_samples.detach()
    # Reactivate model gradients
    model.enable_grads()
    model.train()
    model.disable_batchnorm()
    return trans_samples, negs
