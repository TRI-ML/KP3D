# Copyright 2021 Toyota Research Institute.  All rights reserved.

import torch

def invert_pose(T01):
    """Invert homogeneous matrix as a rigid transformation
       T^-1 = [R^T | -R^T * t]

    Parameters
    ----------
    T01: torch.FloatTensor (B44)
        Input batch of transformation tensors.

    Returns
    ----------
    T10: torch.FloatTensor (B44)
        Inverted batch of transformation tensors.
    """
    Tinv = torch.eye(4, device=T01.device, dtype=T01.dtype).repeat([len(T01), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T01[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3], T01[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv


class Pose:
    """Generic rigid-body transformation class that operates on the
    appropriately defined manifold.

    Parameters
    ----------
    T: torch.FloatTensor (44, B44)
        Input transformation tensors either batched (B44) or as a single value (44).

    Attributes
    ----------
    T: torch.FloatTensor (B44)
        Input transformation tensor batched (B44)
    """
    def __init__(self, value):
        assert tuple(value.shape[-2:]) == (4, 4)
        if value.dim() == 2:
            value = value.unsqueeze(0)
        assert value.dim() == 3
        self.value = value

    def __len__(self):
        """Batch size of pose tensor"""
        return len(self.value)

    def __getitem__(self, index):
        """Allows selecting a specific subset of the batched pose elements.

        Parameters
        ----------
        index: int or slice
            Subset of the batch to select

        Returns
        ----------
        Pose (or subclass)
            Subset of the original batch of Pose objects.

        """
        #  TODO check how this affects gradients.
        return self.__class__(self.value[index])

    @classmethod
    def identity(cls, N=1, device=None, dtype=torch.float):
        """Batch of identity matrices.

        Parameters
        ----------
        N: int
            Batch size.

        Returns
        ----------
        Pose
            Batch of identity transformation poses.
        """
        return cls(torch.eye(4, device=device, dtype=dtype).repeat([N,1,1]))

    @property
    def shape(self):
        return self.value.shape

    def item(self):
        """Return the Pose tensor

        Returns
        ----------
        torch.FloatTensor
            Tensor corresponding to the Pose.
        """
        return self.value

    def repeat(self, *args, **kwargs):
        """Repeat the Pose tensor"""
        self.value = self.value.repeat(*args, **kwargs)
        return self

    def to(self, *args, **kwargs):
        """Move object to specified device"""
        self.value = self.value.to(*args, **kwargs)
        return self

    def __matmul__(self, other):
        """Matrix multiplication overloading for pose-pose and pose-point
        transformations.

        Parameters
        ----------
        other: Pose or torch.FloatTensor
            Either Pose, or 3-D points torch.FloatTensor (B3N or B3HW).

        Returns
        ----------
        Pose
            Transformed pose, or 3-D points via rigid-transform on the manifold,
           with same type as other.
        """
        if isinstance(other, Pose):
            return self.transform_pose(other)
        elif isinstance(other, torch.Tensor):
            if other.shape[1] == 3 and other.dim() > 2:
                assert other.dim() == 3 or other.dim() == 4
                return self.transform_points(other)
            else:
                raise ValueError('Unknown tensor dimensions {}'.format(other.shape))
        else:
            raise NotImplementedError()

    def transform_pose(self, other):
        """Left-multiply (oplus) rigid-body transformation.

        Parameters
        ----------
        other: Pose
            Pose to left-multiply with (self * other)

        Returns
        ----------
        Pose
            Transformed Pose via rigid-transform on the manifold.
        """
        assert tuple(other.shape[-2:]) == (4, 4)
        return Pose(self.value.bmm(other.item()))

    def transform_points(self, X0):
        """Transform 3-D points from one frame to another via rigid-body transformation.

        Parameters
        ----------
        X0: torch.FloatTensor (B3N or B3HW)
            3-D points in torch.FloatTensor (shaped either B3N or B3HW).

        Returns
        ----------
        torch.FloatTensor (B3N or B3HW)
           Transformed 3-D points with the same shape as X0.
        """
        assert X0.shape[1] == 3
        B = len(X0)
        shape = X0.shape[2:]
        X1 = self.value[:,:3,:3].bmm(X0.view(B, 3, -1)) + self.value[:,:3,-1].unsqueeze(-1)
        return X1.view(B, 3, *shape)

    def inverse(self):
        """Invert homogeneous matrix as a rigid transformation.

        Returns
        ----------
        Pose
           Pose batch inverted on the appropriate manifold.
        """
        return Pose(invert_pose(self.value))
