import torch
import collections
import os
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import numpy as np
from networks import conv
from scipy import interpolate

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def apply_disp(img, disp):

    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear', padding_mode='zeros')

    return output


def warp_right(left, disp):
    return apply_disp(left, disp)


def warp_left(right, disp):
    return apply_disp(right, -disp)


def scale_pyramid(img):
    scaled_imgs = [img]
    s = img.size()
    h = s[2]
    w = s[3]
    for i in range(3):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        scaled_imgs.append(F.interpolate(img, size=[nh, nw], mode='bilinear', align_corners=True))
    return scaled_imgs


def depth2disp(depth):
    width, height = depth.shape
    x, y = np.arange(0, height), np.arange(0, width)
    arr = np.ma.masked_equal(depth, 0.0)
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~arr.mask]
    y1 = yy[~arr.mask]
    newarr = arr[~arr.mask]
    inter = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
    return 359.7176277195809831 * 0.54 / inter