# Copyright 2021 Toyota Research Institute.  All rights reserved.

import argparse
import copy
import os
import random
import subprocess

import cv2
import horovod.torch as hvd
import numpy as np
import torch
from evo.core import trajectory
from evo.core.trajectory import PosePath3D
from evo.tools import file_interface
from kp3d.datasets.kitti_odometry_dataset import KITTIODOMETRYDataset
from kp3d.geometry.camera import Camera
from kp3d.geometry.pose import Pose
from kp3d.networks.disp_resnet import DispResnet
from kp3d.networks.keypoint_resnet import KeypointResnet
from kp3d.utils.image import to_color_normalized
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Script for evaluating KP3D keypoint and depth models on the KITTI odometry dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--depth_model', type=str, help='pretrained depth model path')
parser.add_argument('--keypoint_model', type=str, help='pretrained keypoint model path')
parser.add_argument('--dataset_dir', default='.', type=str, help='Dataset directory')
parser.add_argument('--sequence', nargs='*', default=["01"], type=str, help='sequence to test')
parser.add_argument('--output_dir', default='./pose_output/', type=str, help='Output directory for saving depth evaluation results')
parser.add_argument('--align_trajectory', action='store_true', help='Perform scale alignment given predicted and ground truth trajectory')
parser.add_argument('--run_evaluation', action='store_true', help='Run KITTI odometry evaluation')
parser.add_argument('--input_height', help='Input image height', type=int, default=192)
parser.add_argument('--input_width', help='Input image width', type=int, default=640)


def printcolor(message, color="white"):
    "Print a message in a certain color (only rank 0)"
    if hvd.rank() == 0:
        print(colored(message, color))

def match_features(keypoint_net, depth_net, target_image, source_image, target_intrinsic, top_k=480):
        def keep_top_k(score, uv, feat, top_k):
            B, C, Hc, Wc = feat.shape
            # Get top_k keypoint mask
            top_k_score, top_k_indices = score.view(B,Hc*Wc).topk(top_k, dim=1, largest=True)
            top_k_mask = torch.zeros(B, Hc * Wc).to(score.device)
            top_k_mask.scatter_(1, top_k_indices, value=1)
            top_k_mask = top_k_mask.gt(1e-3).view(Hc,Wc)

            # Select top_k keypoints, descriptors and depth
            uv    = uv.squeeze().permute(1,2,0)
            feat  = feat.squeeze().permute(1,2,0)
            top_k_uv    = uv[top_k_mask].view(top_k, 2)
            top_k_feat  = feat[top_k_mask].view(top_k, C)

            return top_k_score, top_k_uv, top_k_feat, top_k_mask

        # Set models to eval
        keypoint_net.eval()
        depth_net.eval()

        # Get dimensions
        B, _, H, W = target_image.shape

        # Extract target and source keypoints, descriptors and score
        target_image = to_color_normalized(target_image.clone())
        target_score, target_uv, target_feat = keypoint_net(target_image)
        source_image = to_color_normalized(source_image.clone())
        ref_score, ref_uv, ref_feat = keypoint_net(source_image)

        # Sample (sparse) target keypoint depth
        target_uv_norm = target_uv.clone()
        target_uv_norm[:,0] = (target_uv_norm[:,0] / (float(W-1)/2.)) - 1.
        target_uv_norm[:,1] = (target_uv_norm[:,1] / (float(H-1)/2.)) - 1.
        target_uv_norm = target_uv_norm.permute(0, 2, 3, 1)

        # Compute target depth
        target_inv_depth = depth_net(target_image)
        target_depth_all = 1 / target_inv_depth.clamp(min=1e-6)
        target_depth = torch.nn.functional.grid_sample(target_depth_all, target_uv_norm.detach(), mode='bilinear')

        # Get top_k source keypoints
        ref_score, ref_uv, ref_feat, _ = keep_top_k(ref_score, ref_uv, ref_feat, top_k=top_k)

        # Get top_k target keypoints
        target_score, target_uv, target_feat, target_top_k_mask = keep_top_k(target_score, target_uv, target_feat, top_k=top_k)

        # Get corresponding target top_k depth
        target_depth = target_depth.squeeze()
        target_depth = target_depth[target_top_k_mask]
        
        # Create target sparse point cloud
        target_cam = Camera(K=target_intrinsic.float()).to(target_image.device)
        target_uvz = torch.cat([target_uv, target_depth.unsqueeze(1)], 1).t().unsqueeze(1)
        target_xyz = target_cam.reconstruct_sparse(target_uvz.view(B,3,-1), frame='c').squeeze().t()

        # Compute descrpitor matches
        target_feat_np = target_feat.t().cpu().numpy()
        ref_feat_np = ref_feat.t().cpu().numpy()

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(target_feat_np.T, ref_feat_np.T)
        target_idx = np.array([m.queryIdx for m in matches])        
        ref_idx = np.array([m.trainIdx for m in matches])
        
        # Create PnP arrays
        pt0 = target_uv[target_idx].cpu().numpy()
        pt1 = target_xyz[target_idx].cpu().numpy()
        pt2 = ref_uv[ref_idx].cpu().numpy()

        # Run PnP and get pose
        K = target_intrinsic.float().numpy().squeeze()
        _, rvecs, t, inliers = cv2.solvePnPRansac(pt1, pt2, K, None, reprojectionError=0.85, iterationsCount=5000)
        R = cv2.Rodrigues(rvecs)[0]
        P = np.concatenate([R,t], 1)

        return P

def accumulate_pose_single_frame(
    dataloader, depth_net, keypoint_net, trajectory_length, rank=0, device='cuda'
):
    """
    Evaluates a model (can be any network which implements the method compute_poses(ref_image, target_images)).
    Also Computes ATE on a snippet of length (back_context+forward_context+1).
    print('T1 positiona shape {} r_a shape {} t_a shape {}'.format(t1_aligned.positions_xyz.shape, r_a.shape, t_a.shape))
    Note: to compute ATE on a snippet of a different length (e.g. 5), run benchmark_pose.sh.
    Parameters
    ----------
    dataloader: torch.dataloader
        Wraps the dataset and returns snippets of appropriate size
    depth_net: torch.nn.Module
        Depth network.
    keypoint_net: torch.nn.Module
        Keypoint network.
    trajectory_length: int
        Length of the entire trajectory.
    rank: int
        Node rank (set by MPI); used to trigger printing.
    device: str
        Device on which to run the computation (e.g. cuda).
    Returns
    ----------
    trajectory: np.array (trajectory_length, 4, 4)
        The computed poses for the complete trajectory.
    """
    all_intermediate_poses = torch.zeros(trajectory_length, 4, 4)
    # aggregate number of times each index is seen (might be twice if dataset size not divisible by
    # world size: DistributedSampler needs same shard size per rank)
    indexes_seen = torch.zeros(trajectory_length)

    # model.eval()
    depth_net.eval()
    keypoint_net.eval()
    with torch.no_grad():
        for sample in tqdm(dataloader, desc='evaluate_pose', disable=(hvd.rank() > 0), unit_scale=hvd.size()):
            j = int(sample['idx'])
            # Get images
            indexes_seen[j] = 1
            batch_size = dataloader.batch_size
            target_image = sample['imgs'][0].to(device)
            source_image = sample['imgs'][1].to(device)
            # Compute pose
            pose_kpt = match_features(keypoint_net, depth_net, target_image, source_image, sample['intrinsic'])
            pose_data = torch.eye(4, dtype=torch.float)
            pose_data[:3, :] = torch.from_numpy(pose_kpt)
            # Store pose
            poses = Pose(pose_data).inverse()
            all_intermediate_poses[j] = poses.item().clone()

    # allreduce the errors/poses (as all ranks compute independent errors and intermediate poses)
    # NB: allreduce is not inplace: gotta assign the return value!)
    all_intermediate_poses = hvd.allreduce(all_intermediate_poses, average=False, name='all_intermediate_poses')
    indexes_seen = hvd.allreduce(indexes_seen, average=False, name='indexes_seen')
    # normalize: non-full last batch and DistributedSampler padding --> some might be seen twice
    # (and some might not be seen at beginning / end due to context)
    unseen = indexes_seen == 0
    indexes_seen[unseen] = 1  # to avoid div by 0 (corresponding numerators are 0)
    all_intermediate_poses /= indexes_seen.view(-1, 1, 1)

    # accumulate intermediate poses for the whole trajectory
    trajectory = torch.zeros(trajectory_length, 4, 4)
    if rank == 0:
        trajectory[0] = torch.eye(4)
        for j in range(0, trajectory_length - 1):
            trajectory[j + 1] = torch.FloatTensor(np.matmul(trajectory[j], all_intermediate_poses[j]))
    trajectory = hvd.allreduce(trajectory, average=False, name='all_poses')
    return trajectory.numpy()

def main():
    hvd.init()
    args = parser.parse_args()

    if hvd.size() > 1:
        printcolor('----------- DISTRIBUTED DATA PARALLEL -----------', 'cyan')
        device_id = hvd.local_rank()
        torch.cuda.set_device(device_id)

    if hvd.rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    depth_net = DispResnet(version='18_pt')
    depth_net.load_state_dict(torch.load(args.depth_model, map_location='cpu'))
    depth_net = depth_net.cuda()  # move to GPU

    keypoint_net = KeypointResnet()
    keypoint_net.load_state_dict(torch.load(args.keypoint_model, map_location='cpu'))
    keypoint_net.cuda()

    def _set_seeds(seed=42):
        """Set Python random seeding and PyTorch seeds.
        Parameters
        ----------
        seed: int, default: 42
            Random number generator seeds for PyTorch and python
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _worker_init_fn(worker_id):
        """Worker init fn to fix the seed of the workers"""
        _set_seeds(42 + worker_id)

    if args.run_evaluation:
        all_trel = []
        all_rrel = []
    ########## STACK PREDICTED POSES INTO FULL TRAJECTORY  ##########
    printcolor('Evaluation sequences {}'.format(args.sequence))

    for seq in args.sequence:
        dataset = KITTIODOMETRYDataset(args.dataset_dir, [seq],
            back_context=0,
            forward_context=1,
            image_shape=(args.input_height, args.input_width),
        )
        trajectory_length = sum(len(imgs) for imgs in dataset.img_files)
        printcolor('Evaluating SEQ {}'.format(seq))

        sampler = None if not (hvd.size() > 1) else \
            torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank())

        dl = DataLoader(
            dataset,
            batch_size=1,
            pin_memory=False,
            shuffle=False,
            num_workers=48,
            worker_init_fn=_worker_init_fn,
            sampler=sampler
        )

        all_poses = accumulate_pose_single_frame(
            dataloader=dl,
            depth_net=depth_net,
            keypoint_net=keypoint_net,
            trajectory_length=trajectory_length,
            rank=hvd.rank()
        )

        if hvd.rank() == 0:
            np.savetxt(
                os.path.join(args.output_dir,
                            str(seq) + '_ori.txt'),
                all_poses.reshape(trajectory_length, -1)[:, :12]
            )
        
        if args.align_trajectory:
            if hvd.rank() == 0:
                printcolor('Computing scale alignment.')
                # Load ground truth poses
                ground_truth_trajectory = file_interface.read_kitti_poses_file(
                    os.path.join(os.path.join(args.dataset_dir, 'poses'), seq) + '.txt'
                )
                # Convert predicted trajectory
                predicted_trajectory = PosePath3D(poses_se3=all_poses)
                # Umeyama alignment with scaling only
                predicted_trajectory_aligned = copy.deepcopy(predicted_trajectory)
                predicted_trajectory_aligned.align(ground_truth_trajectory, correct_only_scale=True)
                # Save aligned trajectory
                file_interface.write_kitti_poses_file(os.path.join(args.output_dir, seq) + '.txt', predicted_trajectory_aligned)

        
        if args.run_evaluation and hvd.rank() == 0:
            pose_test_executable = "./kp3d/externals/cpp/evaluate_odometry"
            p = subprocess.Popen([pose_test_executable, args.dataset_dir, args.output_dir, seq],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
            p.wait()

            seq_output = os.path.join(args.output_dir, seq + '_stats.txt')
            with open(seq_output, 'r') as f:
                pose_result = f.readlines()[0].split(' ')
                pose_result = [float(p) for p in pose_result]

            traj_pos_error = pose_result[0]
            traj_rot_error = pose_result[1]
            all_trel.append(traj_pos_error)
            all_rrel.append(traj_rot_error)


            printcolor('SEQ {} -- Mean TRAJ (pos: {:.4f}, rot: {:.4f})'.format(seq, traj_pos_error, traj_rot_error), 'green')

    printcolor(all_trel)
    printcolor(all_rrel)



if __name__ == '__main__':
    main()
