# Copyright 2021 Toyota Research Institute.  All rights reserved.

# Code adapted in part from:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py

import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
from path import Path
from torch.utils.data import Dataset


class KITTIODOMETRYDataset(Dataset):
    def __init__(self, root, sequence_set, back_context=1, forward_context=1, step=1, image_shape=(192, 640)):
        self.root = root
        self.sequence_set = sequence_set
        self.image_shape = image_shape
        self.image_shape_ori = image_shape
        self.back_context = back_context
        self.forward_context = forward_context
        self.resize_transform = transforms.Resize(self.image_shape, interpolation=Image.BILINEAR)
        self.to_tensor_transform = transforms.ToTensor()
        self.img_files, self.poses, self.sample_indices, self.intrinsics = read_scene_data(self.root,
                                                                          sequence_set=sequence_set,
                                                                          back_context=back_context,
                                                                          forward_context=forward_context,
                                                                          step=step)

    def __getitem__(self, idx):
        img_list, pose_list, sample_list = self.img_files[0], self.poses[0], self.sample_indices[0]

        intrinsic = self.intrinsics[0]
        snippet_indices = sample_list[idx]
        imgs = [Image.open(img_list[i]) for i in snippet_indices]
    
        (orig_w, orig_h) = imgs[0].size        
        (out_h, out_w) = self.image_shape
        y_scale = out_h / orig_h
        x_scale = out_w / orig_w
        intrinsic = np.copy(intrinsic)
        intrinsic[0] *= x_scale
        intrinsic[1] *= y_scale

        imgs = [self.resize_transform(i) for i in imgs]
        # to tensor
        imgs = [self.to_tensor_transform(i).type('torch.FloatTensor') for i in imgs]

        poses = np.stack([pose_list[i] for i in snippet_indices])
        first_pose = poses[0]
        poses[:, :, -1] -= first_pose[:, -1]
        compensated_poses = np.linalg.inv(first_pose[:, :3]) @ poses

        return {'imgs': imgs,
                'poses': compensated_poses,
                'idx': idx,
                'intrinsic': intrinsic,
                'orig_w': orig_w,
                'orig_h': orig_h
                }

    def __len__(self):
        return sum(len(indices) for indices in self.sample_indices)


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def read_scene_data(data_root, sequence_set, back_context=1, forward_context=1, step=1):
    data_root = Path(data_root)
    im_sequences = []
    poses_sequences = []
    indices_sequences = []
    intrinsic_sequences = []
    shift_range = np.array([step * i for i in range(-back_context, forward_context + 1)]).reshape(1, -1)

    sequences = set()
    for seq in sequence_set:
        corresponding_dirs = set((data_root / 'sequences').dirs(seq))
        sequences = sequences | corresponding_dirs

    for sequence in sequences:
        poses = np.genfromtxt(data_root / 'poses' / '{}.txt'.format(sequence.name)).astype(
            np.float64).reshape(-1, 3, 4)
        imgs = sorted((sequence / 'image_2').files('*.png'))
        # construct snippet sequences
        tgt_indices = np.arange(back_context, len(imgs) - forward_context).reshape(-1, 1)
        snippet_indices = shift_range + tgt_indices
        im_sequences.append(imgs)
        poses_sequences.append(poses)
        indices_sequences.append(snippet_indices)

        c_data = read_calib_file(os.path.join(sequence, 'calib.txt'))
        p_rect = np.reshape(c_data['P2'], (3, 4))
        intrinsic_sequences.append(p_rect[:, :3])

    return im_sequences, poses_sequences, indices_sequences, intrinsic_sequences
