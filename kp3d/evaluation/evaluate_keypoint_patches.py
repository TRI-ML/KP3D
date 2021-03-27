# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Example usage: python scripts/eval_keypoint_net.sh --pretrained_model /data/models/kp2d/v4.pth --input_dir /data/datasets/kp2d/HPatches/

import argparse
import os
import pickle
import random
import subprocess

import cv2
import numpy as np
import torch
from kp3d.datasets.patches_dataset import PatchesDataset
from kp3d.evaluation.evaluate import evaluate_keypoint_net
from kp3d.networks.keypoint_resnet import KeypointResnet
from PIL import Image
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointnet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path")
    parser.add_argument("--input_dir", required=True, type=str, help="Folder containing input images")
    args = parser.parse_args()

    keypoint_net = KeypointResnet()
    eval_params = [
            {'res': (320, 256), 'top_k': 300, },
            {'res': (640, 480), 'top_k': 1000, }
        ]

    keypoint_net.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()
    print('Loaded KeypointNet from {}'.format(args.pretrained_model))

    for params in eval_params:
        hp_dataset = PatchesDataset(root_dir=args.input_dir, use_color=True,
                                    output_shape=params['res'], type='a')
        data_loader = DataLoader(hp_dataset,
                                 batch_size=1,
                                 pin_memory=False,
                                 shuffle=False,
                                 num_workers=8,
                                 worker_init_fn=None,
                                 sampler=None)

        print(colored('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']),'green'))
        rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net(
            data_loader,
            keypoint_net,
            output_shape=params['res'],
            top_k=params['top_k'],
            use_color=True)

        print('Repeatability {0:.3f}'.format(rep))
        print('Localization Error {0:.3f}'.format(loc))
        print('Correctness d1 {:.3f}'.format(c1))
        print('Correctness d3 {:.3f}'.format(c3))
        print('Correctness d5 {:.3f}'.format(c5))
        print('MScore {:.3f}'.format(mscore))


if __name__ == '__main__':
    main()
