"""Function common in training/testing of all models."""

import os.path as osp

import torch


@torch.no_grad()
def clip_grad(parameters, optimizer):
    """Clip gradients of given parameters."""
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


def unnormalize_imagenet_rgb(image, device):
    """Unnormalize normalized rgb using imagenet stats."""
    mean_ = torch.as_tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(device)
    std_ = torch.as_tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(device)
    image = (image * std_) + mean_
    return image


def back2color(image):
    """Convert tensor in [0, 1] to color ByteTensor."""
    return (image*255).type(torch.ByteTensor)


def load_from_ckpnt(ckpnt, model, optimizer=None, scheduler=None):
    """Load trained parameters from given checkpoint."""
    start_epoch = 0
    is_trained = False
    if osp.exists(ckpnt):
        checkpoint = torch.load(ckpnt)
        is_trained = checkpoint.get("is_trained", False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if is_trained:
            return model, optimizer, scheduler, start_epoch, is_trained 
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
    return model, optimizer, scheduler, start_epoch, is_trained
