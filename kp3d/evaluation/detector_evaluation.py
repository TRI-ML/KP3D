# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Adapted from: https://github.com/rpautrat/SuperPoint/blob/master/superpoint/evaluations/detector_evaluation.py

import random
from glob import glob
from os import path as osp

import cv2
import numpy as np
from kp3d.utils.keypoints import warp_keypoints


def compute_repeatability(data, keep_k_points=300, distance_thresh=3):
    """
    Compute the repeatability metric between 2 sets of keypoints inside data.

    Parameters
    ----------
    data: dict
        Input dictionary containing:
        image_shape: tuple (H,W)
            Original image shape.
        homography: numpy.ndarray (3,3)
            Ground truth homography.
        prob: numpy.ndarray (N,3)
            Keypoint vector, consisting of (x,y,probability).
        warped_prob: numpy.ndarray (N,3)
            Warped keypoint vector, consisting of (x,y,probability).
    keep_k_points: int
        Number of keypoints to select, based on probability.
    distance_thresh: int
        Distance threshold in pixels for a corresponding keypoint to be considered a correct match.

    Returns
    -------    
    N1: int
        Number of true keypoints in the first image.
    N2: int
        Number of true keypoints in the second image.
    repeatability: float
        Keypoint repeatability metric.
    loc_err: float
        Keypoint localization error.
    """
    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are inside the dimensions of shape. """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
               (points[:, 1] >= 0) & (points[:, 1] < shape[1])
        return points[mask, :]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H are still inside shape. """
        warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]


    def select_k_best(points, k):
        """ Select the k most probable points (and strip their probability).
        points has shape (num_points, 3) where the last coordinate is the probability. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    H = data['homography']
    shape = data['image_shape']

    # # Filter out predictions
    keypoints = data['prob'][:, :2].T
    keypoints = keypoints[::-1]
    prob = data['prob'][:, 2]

    warped_keypoints = data['warped_prob'][:, :2].T
    warped_keypoints = warped_keypoints[::-1]
    warped_prob = data['warped_prob'][:, 2]

    keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
    warped_keypoints = np.stack([warped_keypoints[0], warped_keypoints[1], warped_prob], axis=-1)
    warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H), shape)

    # Warp the original keypoints with the true homography
    true_warped_keypoints = warp_keypoints(keypoints[:, [1, 0]], H)
    true_warped_keypoints = np.stack([true_warped_keypoints[:, 1], true_warped_keypoints[:, 0], prob], axis=-1)
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

    # Keep only the keep_k_points best predictions
    warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

    # Compute the repeatability
    N1 = true_warped_keypoints.shape[0]
    N2 = warped_keypoints.shape[0]
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)
    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)
    count1 = 0
    count2 = 0
    le1 = 0
    le2 = 0
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        correct1 = (min1 <= distance_thresh)
        count1 = np.sum(correct1)
        le1 = min1[correct1].sum()
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        correct2 = (min2 <= distance_thresh)
        count2 = np.sum(correct2)
        le2 = min2[correct2].sum()
    if N1 + N2 > 0:
        repeatability = (count1 + count2) / (N1 + N2)
        loc_err = (le1 + le2) / (count1 + count2)
    else:
        repeatability = -1
        loc_err = -1

    return N1, N2, repeatability, loc_err
