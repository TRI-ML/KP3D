# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Adapted from: https://github.com/rpautrat/SuperPoint/blob/master/superpoint/evaluations/descriptor_evaluation.py

import random
from glob import glob
from os import path as osp

import cv2
import numpy as np
from kp3d.utils.keypoints import warp_keypoints


def select_k_best(points, descriptors, k):
    """ Select the k most probable points (and strip their probability).
    points has shape (num_points, 3) where the last coordinate is the probability.

    Parameters
    ----------
    points: numpy.ndarray (N,3)
        Keypoint vector, consisting of (x,y,probability).
    descriptors: numpy.ndarray (N,256)
        Keypoint descriptors.
    k: int
        Number of keypoints to select, based on probability.
    Returns
    -------
    
    selected_points: numpy.ndarray (k,2)
        k most probable keypoints.
    selected_descriptors: numpy.ndarray (k,256)
        Descriptors corresponding to the k most probable keypoints.
    """
    sorted_prob = points[points[:, 2].argsort(), :2]
    sorted_desc = descriptors[points[:, 2].argsort(), :]
    start = min(k, points.shape[0])
    selected_points = sorted_prob[-start:, :]
    selected_descriptors = sorted_desc[-start:, :]
    return selected_points, selected_descriptors


def keep_shared_points(keypoints, descriptors, H, shape, keep_k_points=1000):
    """
    Compute a list of keypoints from the map, filter the list of points by keeping
    only the points that once mapped by H are still inside the shape of the map
    and keep at most 'keep_k_points' keypoints in the image.
    
    Parameters
    ----------
    keypoints: numpy.ndarray (N,3)
        Keypoint vector, consisting of (x,y,probability).
    descriptors: numpy.ndarray (N,256)
        Keypoint descriptors.
    H: numpy.ndarray (3,3)
        Homography.
    shape: tuple 
        Image shape.
    keep_k_points: int
        Number of keypoints to select, based on probability.

    Returns
    -------    
    selected_points: numpy.ndarray (k,2)
        k most probable keypoints.
    selected_descriptors: numpy.ndarray (k,256)
        Descriptors corresponding to the k most probable keypoints.
    """
    
    def keep_true_keypoints(points, descriptors, H, shape):
        """ Keep only the points whose warped coordinates by H are still inside shape. """
        # warped_points = warp_keypoints(points[:, [1, 0]], H)[:,[1,0]]
        # warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        warped_points = warp_keypoints(points[:,:2], H)
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :], descriptors[mask, :]

    selected_keypoints, selected_descriptors = keep_true_keypoints(keypoints, descriptors, H, shape)
    selected_keypoints, selected_descriptors = select_k_best(selected_keypoints, selected_descriptors, keep_k_points)
    return selected_keypoints, selected_descriptors


def compute_matching_score(data, keep_k_points=1000):
    """
    Compute the matching score between two sets of keypoints with associated descriptors.
    
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
        desc: numpy.ndarray (N,256)
            Keypoint descriptors.
        warped_desc: numpy.ndarray (N,256)
            Warped keypoint descriptors.
    keep_k_points: int
        Number of keypoints to select, based on probability.

    Returns
    -------    
    ms: float
        Matching score.
    """
    shape = data['image_shape']
    real_H = data['homography']

    # Filter out predictions
    keypoints = data['prob']
    warped_keypoints = data['warped_prob']

    desc = data['desc']
    warped_desc = data['warped_desc']
    
    # Keeps all points for the next frame. The matching for caculating M.Score shouldnt use only in view points.
    keypoints,        desc        = select_k_best(keypoints,               desc, keep_k_points)
    warped_keypoints, warped_desc = select_k_best(warped_keypoints, warped_desc, keep_k_points)
    
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    # This part needs to be done with crossCheck=False.
    # All the matched pairs need to be evaluated without any selection.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches = bf.match(desc, warped_desc)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_keypoints = keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]

    true_warped_keypoints = warp_keypoints(m_warped_keypoints, np.linalg.inv(real_H))
    vis_warped = np.all((true_warped_keypoints >= 0) & (true_warped_keypoints <= (np.array(shape)-1)), axis=-1)
    norm1 = np.linalg.norm(true_warped_keypoints - m_keypoints, axis=-1)

    correct1 = (norm1 < 3)
    count1 = np.sum(correct1 * vis_warped)
    score1 = count1 / np.maximum(np.sum(vis_warped), 1.0)

    matches = bf.match(warped_desc, desc)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_keypoints = keypoints[matches_idx, :]

    true_keypoints = warp_keypoints(m_keypoints, real_H)
    vis = np.all((true_keypoints >= 0) & (true_keypoints <= (np.array(shape)-1)), axis=-1)
    norm2 = np.linalg.norm(true_keypoints - m_warped_keypoints, axis=-1)

    correct2 = (norm2 < 3)
    count2 = np.sum(correct2 * vis)
    score2 = count2 / np.maximum(np.sum(vis), 1.0)

    ms = (score1 + score2) / 2

    return ms

def compute_homography(data, keep_k_points=1000):
    """
    Compute the homography between 2 sets of Keypoints and descriptors inside data. 
    Use the homography to compute the correctness metrics (1,3,5).

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
        desc: numpy.ndarray (N,256)
            Keypoint descriptors.
        warped_desc: numpy.ndarray (N,256)
            Warped keypoint descriptors.
    keep_k_points: int
        Number of keypoints to select, based on probability.

    Returns
    -------    
    correctness1: float
        correctness1 metric.
    correctness3: float
        correctness3 metric.
    correctness5: float
        correctness5 metric.
    """
    shape = data['image_shape'][::-1]
    real_H = data['homography']

    keypoints = data['prob']
    warped_keypoints = data['warped_prob']

    desc = data['desc']
    warped_desc = data['warped_desc']
    
    # Keeps only the points shared between the two views
    keypoints, desc = keep_shared_points(keypoints, desc, real_H, shape, keep_k_points)
    warped_keypoints, warped_desc = keep_shared_points(warped_keypoints, warped_desc, np.linalg.inv(real_H), shape,
                                                       keep_k_points)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc, warped_desc)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_keypoints = keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]

    # Estimate the homography between the matches using RANSAC
    H, _ = cv2.findHomography(m_keypoints, m_warped_keypoints, cv2.RANSAC, 3, maxIters=5000)

    if H is None:
        return 0, 0, 0

    # Compute correctness
    corners = np.array([[0, 0, 1],
                        [0, shape[1] - 1, 1],
                        [shape[0] - 1, 0, 1],
                        [shape[0] - 1, shape[1] - 1, 1]])
    real_warped_corners = np.dot(corners, np.transpose(real_H))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(H))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    
    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    correctness1 = float(mean_dist <= 1)
    correctness3 = float(mean_dist <= 3)
    correctness5 = float(mean_dist <= 5)

    return correctness1, correctness3, correctness5
