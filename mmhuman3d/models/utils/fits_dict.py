# ------------------------------------------------------------------------------
# Adapted from https://github.com/nkolot/SPIN/blob/master/train/fits_dict.py
# Original licence please see docs/additional_licenses.md
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import torch

from mmhuman3d.utils.transforms import aa_to_rotmat

train_datasets = ['h36m', 'mpi_inf_3dhp', 'lsp', 'lspet', 'mpii', 'coco']
static_fits_load_dir = 'data/static_fits'
save_dir = 'data/spin_fits'

# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [
    0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21,
    20, 23, 22
]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3 * i)
    SMPL_POSE_FLIP_PERM.append(3 * i + 1)
    SMPL_POSE_FLIP_PERM.append(3 * i + 2)


class FitsDict():
    """Dictionary keeping track of the best fit per image in the training set.

    Ref: https://github.com/nkolot/SPIN/blob/master/train/fits_dict.py
    """

    def __init__(self, fits='static') -> None:
        assert fits in ['static', 'final']
        self.fits = fits
        self.fits_dict = {}

        # array used to flip SMPL pose parameters
        self.flipped_parts = torch.tensor(
            SMPL_POSE_FLIP_PERM, dtype=torch.int64)
        # Load dictionary state
        # for ds_name, ds in train_dataset.dataset_dict.items():
        for ds_name in train_datasets:

            # h36m has gt so no static fits
            if ds_name == 'h36m' or self.fits == 'static':
                dict_file = os.path.join(static_fits_load_dir,
                                         ds_name + '_fits.npy')
                content = np.load(dict_file)
                self.fits_dict[ds_name] = torch.from_numpy(content)
                del content
            elif self.fits == 'final':
                dict_file = os.path.join('data/final_fits', ds_name + '.npz')
                # load like this to save mem
                content = np.load(dict_file)
                pose = torch.from_numpy(content['pose'])
                betas = torch.from_numpy(content['betas'])
                del content
                params = torch.cat([pose, betas], dim=-1)
                self.fits_dict[ds_name] = params

    def save(self):
        """Save dictionary state to disk."""
        for ds_name in train_datasets:
            dict_file = os.path.join(save_dir, ds_name + '_fits.npy')
            np.save(dict_file, self.fits_dict[ds_name].cpu().numpy())

    def __getitem__(self, x):
        """Retrieve dictionary entries."""
        dataset_name, ind, rot, is_flipped = x
        batch_size = len(dataset_name)
        pose = torch.zeros((batch_size, 72))
        betas = torch.zeros((batch_size, 10))
        for ds, i, n in zip(dataset_name, ind, range(batch_size)):
            params = self.fits_dict[ds][i]
            pose[n, :] = params[:72]
            betas[n, :] = params[72:]
        pose = pose.clone()

        # Apply flipping and rotation
        pose = self.rotate_pose(self.flip_pose(pose, is_flipped), rot)

        betas = betas.clone()
        return pose, betas

    def __setitem__(self, x, val):
        """Update dictionary entries."""
        dataset_name, ind, rot, is_flipped, update = x
        pose, betas = val
        batch_size = len(dataset_name)

        # Undo flipping and rotation
        pose = self.flip_pose(self.rotate_pose(pose, -rot), is_flipped)

        params = torch.cat((pose, betas), dim=-1).cpu()
        for ds, i, n in zip(dataset_name, ind, range(batch_size)):
            if update[n]:
                self.fits_dict[ds][i] = params[n]

    def flip_pose(self, pose, is_flipped):
        """flip SMPL pose parameters."""
        is_flipped = is_flipped.bool()
        pose_f = pose.clone()
        pose_f[is_flipped, :] = pose[is_flipped][:, self.flipped_parts]
        # we also negate the second and the third dimension of the
        # axis-angle representation
        pose_f[is_flipped, 1::3] *= -1
        pose_f[is_flipped, 2::3] *= -1
        return pose_f

    def rotate_pose(self, pose, rot):
        """Rotate SMPL pose parameters by rot degrees."""
        pose = pose.clone()
        cos = torch.cos(-np.pi * rot / 180.)
        sin = torch.sin(-np.pi * rot / 180.)
        zeros = torch.zeros_like(cos)
        r3 = torch.zeros(cos.shape[0], 1, 3, device=cos.device)
        r3[:, 0, -1] = 1
        R = torch.cat([
            torch.stack([cos, -sin, zeros], dim=-1).unsqueeze(1),
            torch.stack([sin, cos, zeros], dim=-1).unsqueeze(1), r3
        ],
                      dim=1)
        global_pose = pose[:, :3]
        global_pose_rotmat = R @ aa_to_rotmat(global_pose)
        global_pose_rotmat = global_pose_rotmat.cpu().numpy()
        global_pose_np = np.zeros((global_pose.shape[0], 3))
        for i in range(global_pose.shape[0]):
            aa, _ = cv2.Rodrigues(global_pose_rotmat[i])
            global_pose_np[i, :] = aa.squeeze()
        pose[:, :3] = torch.from_numpy(global_pose_np).to(pose.device)
        return pose
