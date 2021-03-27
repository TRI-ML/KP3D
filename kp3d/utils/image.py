# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=None)
def meshgrid(B, H, W, dtype, device, normalized=False):
    """Create mesh-grid given batch size, height and width dimensions.

    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: torch.dtype
        Tensor dtype
    device: str
        Tensor device
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    xs: torch.Tensor
        Batched mesh-grid x-coordinates (BHW).
    ys: torch.Tensor
        Batched mesh-grid y-coordinates (BHW).
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])


@lru_cache(maxsize=None)
def image_grid(B, H, W, dtype, device, ones=True, normalized=False):
    """Create an image mesh grid with shape B3HW given image shape BHW

    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: str
        Tensor dtype
    device: str
        Tensor device
    ones : bool
        Use (x, y, 1) coordinates
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    grid: torch.Tensor
        Mesh-grid for the corresponding image shape (B3HW)
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    coords = [xs, ys]
    if ones:
        coords.append(torch.ones_like(xs))  # BHW
    grid = torch.stack(coords, dim=1)  # B3HW
    return grid


def to_gray_normalized(images):
    """Performs image normalization and converts images to grayscale (preserving dimensions)
    
    Parameters
    ----------
    images: torch.Tensor
        Input images.

    Returns
    -------
    normalized_images: torch.Tensor
        Normalized grayscale images.
    """
    assert len(images.shape) == 4
    images -= 0.5
    images *= 0.225
    normalized_images = images.mean(1).unsqueeze(1) 
    return normalized_images


def to_color_normalized(images):
    """Performs image normalization and converts images to grayscale (preserving dimensions)
    
    Parameters
    ----------
    images: torch.Tensor
        Input images.

    Returns
    -------
    normalized_images: torch.Tensor
        Normalized grayscale images.
    """
    assert len(images.shape) == 4
    images -= 0.5
    images *= 0.225
    return images
