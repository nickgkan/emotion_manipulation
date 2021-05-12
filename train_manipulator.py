"""Train Manipulator / Inpainter"""

import pkbar
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from train_test_utils import (
    load_from_ckpnt, clip_grad, back2color, unnormalize_imagenet_rgb, rand_mask
)

import ipdb
st = ipdb.set_trace


def train_manipulator(model, data_loaders, args):
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
            neg_samples, neg_masks = rand_mask(images.clone().to(device), device)
            # negative samples
            neg_ld_samples, neg_list = langevin_updates(
                model, torch.clone(neg_samples),
                args.langevin_steps, args.langevin_step_size,
                neg_masks
            )
            # Compute energy
            pos_out = model(pos_samples)
            neg_img_out = model(neg_images.to(device))
            neg_ld_out = model(neg_ld_samples.to(device))
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
                'random_image_sample',
                back2color(unnormalize_imagenet_rgb(pos_samples[0], device)),
                epoch * len(data_loaders['train']) + step
            )
            neg_list = [
                back2color(unnormalize_imagenet_rgb(neg, device))
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
            "manipulator_%02d.pt" % (epoch+1)
        )
        print('\nValidation')
        print(eval_manipulator(model, data_loaders['test'], args))
    return model


@torch.no_grad()
def eval_manipulator(model, data_loader, args):
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
        pos_out = model(images.to(device))
        # negative samples
        neg_samples, neg_masks = rand_mask(images.clone().to(device), device)
        neg_img_out = model(neg_samples.to(device))
        gt += len(images)
        pred += (pos_out < neg_img_out).sum()
        kbar.update(step, [("acc", pred / gt)])

    print(f"\nAccuracy: {pred / gt}")
    return pred / gt


def langevin_updates(model, neg_samples, nsteps, langevin_lr, masks=None):
    """Apply nsteps iterations of Langevin dynamics on neg_samples."""
    # Deactivate model gradients
    model.disable_all_grads()
    model.eval()
    model.disable_batchnorm()
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
        if masks is not None:
            neg_samples.data.add_(neg_samples.grad.data * masks, alpha=-langevin_lr)
        else:
            neg_samples.data.add_(neg_samples.grad.data, alpha=-langevin_lr)
        # Zero gradients
        neg_samples.grad.detach_()
        neg_samples.grad.zero_()
        # Clamp
        neg_samples.data.clamp_(-2.5, 2.5) # neg_samples.data.clamp(-0.485 / 0.229, (1 - 0.406) / 0.225)
        # Store intermediate results for visualization
        negs.append(torch.clone(neg_samples[0]).detach())
    # Detach samples
    neg_samples = neg_samples.detach()
    # Reactivate model gradients
    model.enable_grads()
    model.train()
    model.disable_batchnorm()
    return neg_samples, negs
