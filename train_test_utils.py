"""Function common in training/testing of all models."""

import os.path as osp
import numpy as np
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


def normalize_imagenet_rgb(image):
    """Normalize rgb using imagenet stats."""
    mean_ = torch.as_tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(image.device)
    std_ = torch.as_tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(image.device)
    if len(image.shape) == 4:
        mean_ = mean_.unsqueeze(0)
        std_ = std_.unsqueeze(0)
    image = (image - mean_) / std_
    return image


def unnormalize_imagenet_rgb(image):
    """Unnormalize normalized rgb using imagenet stats."""
    mean_ = torch.as_tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(image.device)
    std_ = torch.as_tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(image.device)
    image = (image * std_) + mean_
    return image.clamp(0, 1)


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


def rand_mask(image, device):
    B, C, H, W = image.shape
    crop_h = np.random.randint(int(H/4), int(3*H/4))
    crop_w = np.random.randint(int(W/4), int(3*W/4))
    crop_h_start = np.random.randint(H - crop_h)
    crop_w_start = np.random.randint(W - crop_w)
    crop_h_end = crop_h_start + crop_h
    crop_w_end = crop_w_start + crop_w
    masks = torch.zeros((B,1,H,W)).to(device)
    masks[:,:,crop_h_start:crop_h_end+1,crop_w_start:crop_w_end+1] = 1
    image[:,:,crop_h_start:crop_h_end+1,crop_w_start:crop_w_end+1] = torch.randn_like(image[:,:,crop_h_start:crop_h_end+1,crop_w_start:crop_w_end+1]).to(device)
    return image, masks

def random_brightness(image, coeff):
    # coeff: (B,)
    B, _, H, W = image.shape
    return (image * coeff.reshape(B, 1, 1, 1)).clamp(0, 1)

def random_contrast(image, coeff):
    # coeff: (B,)
    B, _, H, W = image.shape
    return (coeff.reshape(B, 1, 1, 1) * image + (1 - coeff.reshape(B, 1, 1, 1)) * image.mean()).clamp(0,1)

def random_saturation(image, coeff):
    # coeff: (B,)
    B, _, H, W = image.shape
    grayscale_vec = torch.as_tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1).float().cuda()
    return (coeff.reshape(B, 1, 1, 1) * image + (1 - coeff.reshape(B, 1, 1, 1)) * (grayscale_vec * image).sum(1).unsqueeze(1)).clamp(0, 1)

def random_linear(image, w, b):
    # image: (B, 3, 64, 64)
    # w: (B, 3, 3)
    # b: (B, 3)
    B, _, H, W = image.shape
    return torch.bmm(w, image.reshape(B, 3, -1)).reshape(B, 3, H, W) + b.reshape(B, 3, 1, 1)

def rand_augment(image):
    # image: (B, 3, 64, 64)
    B, _, H, W = image.shape
    brightness_params = (torch.ones((B)).float() * 0.8 + torch.rand(B) * 0.4).to(image.device)
    contrast_params = (torch.ones((B)).float() * 0.8 + torch.rand(B) * 0.4).to(image.device)
    saturation_params = (torch.ones((B)).float() * 0.8 + torch.rand(B) * 0.4).to(image.device)
    linear_params_w = (torch.eye(3).unsqueeze(0).repeat(B,1,1) * 0.9 + torch.rand((B,3,3)) * 0.2).to(image.device)
    linear_params_b = (torch.zeros((B, 3)).float() * -0.1 + torch.rand((B,3)) * 0.2).to(image.device)

    image = random_linear(image, linear_params_w, linear_params_b)
    image = random_brightness(image, brightness_params)
    image = random_contrast(image, contrast_params)
    image = random_saturation(image, saturation_params)

    return image
