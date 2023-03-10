import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np



# --------------------------------------------------------------------------------- Train Loss


def matting_loss(pred_fgr, pred_pha, true_fgr, true_pha):
    """
    Args:
        pred_fgr: Shape(B, T, 3, H, W)
        pred_pha: Shape(B, T, 1, H, W)
        true_fgr: Shape(B, T, 3, H, W)
        true_pha: Shape(B, T, 1, H, W)
    """
    loss = dict()
    # Alpha losses
    loss['pha_l1'] = F.l1_loss(pred_pha, true_pha)
    loss['pha_laplacian'] = laplacian_loss(pred_pha.flatten(0, 1), true_pha.flatten(0, 1))
    loss['pha_coherence'] = F.mse_loss(pred_pha[:, 1:] - pred_pha[:, :-1],
                                       true_pha[:, 1:] - true_pha[:, :-1]) * 5

    # Foreground losses
    true_msk = true_pha.gt(0)
    pred_fgr = pred_fgr * true_msk
    true_fgr = true_fgr * true_msk
    loss['fgr_l1'] = F.l1_loss(pred_fgr, true_fgr)
    loss['fgr_coherence'] = F.mse_loss(pred_fgr[:, 1:] - pred_fgr[:, :-1],
                                       true_fgr[:, 1:] - true_fgr[:, :-1]) * 5

    # Total
    loss['total'] = loss['pha_l1'] + loss['pha_coherence'] + loss['pha_laplacian'] \
                             + loss['fgr_l1'] + loss['fgr_coherence']
    return loss

def erode(bin_img, ksize=3):
        #先为原图加入 padding，防止腐蚀后图像尺寸缩小
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2#填充方向设置
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)#对图像的上下左右四个方向填充0
        # 将原图 unfold 成 patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # 取每个 patch 中最小的值
        eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
        return eroded

def dilation(img,kernel_size = 3):
    return nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)(img)

def get_edge(pred_pha, true_pha):
    """
     pred_pha: Shape(B, T, 1, H, W)
     true_pha: Shape(B, T, 1, H, W)
    """
    T_pha = true_pha.flatten(0,1)
    P_pha = pred_pha.flatten(0,1)
    T_erode = 1 - dilation(1 - T_pha)
    T_edge_pre = ((T_pha - T_erode) > 0.001).long()

    T_edge = T_edge_pre * T_pha
    P_dege = T_edge_pre * P_pha
    # for i in range(B):
    #     a = T_edge[i].permute(1,2,0).cpu().numpy()
    #     plt.imshow(a)
    #     plt.savefig(f'/home/xjm/code/RVM/ceshi_img/{i}_a.png')
    #     b = P_dege[i].permute(1,2,0).cpu().numpy()
    #     plt.imshow(b)
    #     plt.savefig(f'/home/xjm/code/RVM/ceshi_img/{i}_b.png')
    return P_dege,T_edge

def matting_pha_edge_loss(pred_pha, true_pha):
    P_dege,T_edge = get_edge(pred_pha, true_pha)
    loss = dict()
    loss['pha_edge'] = F.l1_loss(P_dege,T_edge)
    return loss, P_dege, T_edge


def matting_pha_loss(pred_pha, true_pha):
    """
    Args:
        pred_fgr: Shape(B, T, 3, H, W)
        pred_pha: Shape(B, T, 1, H, W)
        true_fgr: Shape(B, T, 3, H, W)
        true_pha: Shape(B, T, 1, H, W)
    """
    loss = dict()
    # Alpha losses
    loss['pha_l1'] = F.l1_loss(pred_pha, true_pha)
    loss['pha_laplacian'] = laplacian_loss(pred_pha.flatten(0, 1), true_pha.flatten(0, 1))
    loss['pha_coherence'] = F.mse_loss(pred_pha[:, 1:] - pred_pha[:, :-1],
                                       true_pha[:, 1:] - true_pha[:, :-1]) * 5

    # Total
    loss['pha_total_loss'] = loss['pha_l1'] + loss['pha_coherence'] + loss['pha_laplacian']
    return loss

def matting_fgr_loss(pred_fgr, true_fgr, true_pha):
    """
    Args:
        pred_fgr: Shape(B, T, 3, H, W)
        pred_pha: Shape(B, T, 1, H, W)
        true_fgr: Shape(B, T, 3, H, W)
        true_pha: Shape(B, T, 1, H, W)
    """
    loss = dict()
    # Foreground losses
    true_msk = true_pha.gt(0)
    pred_fgr = pred_fgr * true_msk
    true_fgr = true_fgr * true_msk
    loss['fgr_l1'] = F.l1_loss(pred_fgr, true_fgr)
    loss['fgr_coherence'] = F.mse_loss(pred_fgr[:, 1:] - pred_fgr[:, :-1],
                                       true_fgr[:, 1:] - true_fgr[:, :-1]) * 5
    # Total
    loss['fgr_total_loss'] = loss['fgr_l1'] + loss['fgr_coherence']

    return loss

def segmentation_loss(pred_seg, true_seg):
    """
    Args:
        pred_seg: Shape(B, T, 1, H, W)
        true_seg: Shape(B, T, 1, H, W)
    """
    return F.binary_cross_entropy_with_logits(pred_seg, true_seg)


# ----------------------------------------------------------------------------- Laplacian Loss


def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]

