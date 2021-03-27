# Copyright 2021 Toyota Research Institute.  All rights reserved.

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kp3d.geometry.pose import Pose
from kp3d.utils.image import image_grid


def construct_K(fx, fy, cx, cy, dtype=torch.float, device=None):
    """Create camera intrinsics from focal length and focal center

    Parameters
    ----------
    fx: float
        Camera focal length along x-axis.
    fy: float
        Camera focal length along y-axis.
    cx: float
        Camera x-axis principal point.
    cy: float
        Camera y-axis principal point.
    dtype: str
        Tensor dtype
    device: str
        Tensor device

    Returns
    -------
    torch.FloatTensor
        Camera intrinsic matrix (33)
    """
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype, device=device)


def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsic matrix (B33, or 33) given x and y-axes scales.
    Note: This function works for both torch and numpy.

    Parameters
    ----------
    x_scale: float
        x-axis scale factor.
    y_scale: float
        y-axis scale factor.

    Returns
    -------
    torch.FloatTensor (33 or B33)
        Scaled camera intrinsic matrix
    """
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K


class Camera(nn.Module):
    """Fully-differentiable camera class whose extrinsics operate on the
    appropriate pose manifold. Supports fully-differentiable 3D-to-2D, 2D-to-3D
    projection/back-projections, scaled camera and inverse warp
    functionality.

    Note: This class implements the batched camera class, where a batch of
    camera intrinsics (K) and extrinsics (Tcw) are used for camera projection,
    back-projection.

    Attributes
    ----------
    K: torch.FloatTensor (B33)
        Camera calibration matrix.
    Tcw: Pose
        Pose from world to camera frame.
    """

    def __init__(self, K, Tcw=None):
        super(Camera, self).__init__()
        self.K = K
        self.Tcw = Pose.identity(len(K)) \
                   if Tcw is None else Tcw
        assert len(self.Tcw) == len(self.K)
        assert issubclass(type(self.Tcw), Pose), 'Tcw needs to be a Pose type'

    def __len__(self):
        return len(self.K)

    def to(self, *args, **kwargs):
        self.K = self.K.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

    @property
    @lru_cache()
    def Twc(self):
        """Invert extrinsics as a rigid transformation.
           Twc = [Rcw^T | -Rcw^T * tcw]

        Returns
        -------
        Pose
            Inverted batch of poses.
        """
        return self.Tcw.inverse()

    @classmethod
    def from_params(cls, fx, fy, cx, cy, Tcw=None, B=1):
        """Create camera batch from calibration parameters.

        Parameters
        ----------
        fx: float
            Camera focal length along x-axis.
        fy: float
            Camera focal length along y-axis.
        cx: float
            Camera x-axis principal point.
        cy: float
            Camera y-axis principal point.
        Tcw: Pose
            Pose from world to camera frame, with a batch size of 1.
        B: int
            Batch size for Tcw and K

        Returns
        ----------
        Camera
            Camera object with relevant intrinsics and batch size of B.
        """
        if Tcw is not None:
            assert issubclass(type(Tcw), Pose), 'Tcw needs to be a Pose type'
            assert len(Tcw) == 1
            Tcw = Tcw.repeat([B, 1, 1])
        return cls(K=construct_K(fx, fy, cx, cy).repeat([B, 1, 1]), Tcw=Tcw)

    def scaled(self, x_scale, y_scale=None):
        """Scale the camera by specified factor.

        Parameters
        ----------
        x_scale: float
            x-axis scale factor.
        y_scale: float
            y-axis scale factor.

        Returns
        ----------
        Camera
            Scaled camera object.
        """
        if y_scale is None:
            y_scale = x_scale
        if x_scale == 1. and y_scale == 1.:
            return self
        K = scale_intrinsics(self.K.clone(), x_scale, y_scale)
        return Camera(K, Tcw=self.Tcw)

    @property
    def fx(self):
        return self.K[:, 0, 0]

    @property
    def fy(self):
        return self.K[:, 1, 1]

    @property
    def cx(self):
        return self.K[:, 0, 2]

    @property
    def cy(self):
        return self.K[:, 1, 2]

    @property
    @lru_cache()
    def Kinv(self):
        """Analytic inverse camera intrinsic (K^-1)

        Returns
        ----------
        Kinv: torch.FloatTensor (B33)
            Batch of inverse camera matrices K^-1.
        """
        assert tuple(self.K.shape[-2:]) == (3, 3)
        Kinv = self.K.clone()
        Kinv[:, 0, 0] = 1. / self.fx
        Kinv[:, 1, 1] = 1. / self.fy
        Kinv[:, 0, 2] = -1. * self.cx / self.fx
        Kinv[:, 1, 2] = -1. * self.cy / self.fy
        return Kinv

    def reconstruct_sparse(self, uv_depth, frame='c'):
        """Back-project to 3D in specified reference frame, given sparse depth map and uv coordinates

        Parameters
        ----------
        uv_depth: torch.FloatTensor (B3N)
            Depth point cloud, represented as (u,v,d).
        frame: str
            Reference frame in which the output 3-D points are specified.

        Returns
        ----------
        X: torch.FloatTensor (B3HW)
            Batch of 3D spatial coordinates for each pixel in the specified
            reference frame.
        """

        B, C, _ = uv_depth.shape
        assert C == 3

        coords = [uv_depth[:,0], uv_depth[:,1]]
        depth = uv_depth[:,2].unsqueeze(1)
        coords.append(torch.ones_like(uv_depth[:,0])) # BHW
        flat_grid = torch.stack(coords, dim=1) # B3HW

        # Estimate the outward rays in the camera frame
        xnorm = self.Kinv.bmm(flat_grid)

        # Scale rays to metric depth
        Xc = xnorm * depth

        if frame == 'c':
            return Xc
        elif frame == 'w':
            return self.Twc @ Xc
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))
    