import argparse
import glob
import os
import os.path as osp
# import pdb
import pickle

import numpy as np
import torch
# import smplx
# import open3d as o3d
import trimesh
from torch.nn.functional import l1_loss, mse_loss
from tqdm import tqdm

from mmhuman3d.models.body_models.builder import build_body_model


def fit_params(target_verts, body_model, args):

    device = args.device

    target_verts = torch.tensor(
        target_verts, device=device, dtype=torch.float32)
    batch_size = target_verts.shape[0]

    global_orient = torch.zeros((batch_size, 3)).to(device)
    transl = torch.zeros((batch_size, 3)).to(device)
    body_pose = torch.zeros((batch_size, 63)).to(device)
    betas = torch.zeros(batch_size, 10).to(device)
    left_hand_pose = torch.zeros(batch_size, 45).to(device)
    right_hand_pose = torch.zeros(batch_size, 45).to(device)
    jaw_pose = torch.zeros(batch_size, 3).to(device)
    expression = torch.zeros(batch_size, 10).to(device)

    face_vertex_ids = args.face_vertex_ids
    left_hand_vertex_ids = args.left_hand_vertex_ids
    right_hand_vertex_ids = args.right_hand_vertex_ids

    #############################################
    # stage 1: fit global orient and transl only
    #############################################
    optimizer = torch.optim.LBFGS([
        global_orient, transl, body_pose, betas, left_hand_pose,
        right_hand_pose, jaw_pose, expression
    ],
                                  max_iter=20,
                                  lr=1e-2,
                                  line_search_fn='strong_wolfe')

    global_orient.requires_grad = True
    transl.requires_grad = True
    body_pose.requires_grad = False
    betas.requires_grad = False
    left_hand_pose.requires_grad = False
    right_hand_pose.requires_grad = False
    jaw_pose.requires_grad = False
    expression.requires_grad = False

    max_iter = 20
    face_extra_weight = 0.0
    hand_extra_weight = 0.0
    for i in range(max_iter):

        def closure():
            optimizer.zero_grad()

            output = body_model(
                global_orient=global_orient,
                transl=transl,
                body_pose=body_pose,
                betas=betas,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=jaw_pose,
                expression=expression,
                return_verts=True)
            pred_verts = output['vertices']
            face_loss = mse_loss(pred_verts[:, face_vertex_ids],
                                 target_verts[:, face_vertex_ids])
            left_hand_loss = l1_loss(pred_verts[:, left_hand_vertex_ids],
                                     target_verts[:, left_hand_vertex_ids])
            right_hand_loss = l1_loss(pred_verts[:, right_hand_vertex_ids],
                                      target_verts[:, right_hand_vertex_ids])
            all_loss = mse_loss(pred_verts, target_verts)
            loss = all_loss + face_extra_weight * face_loss
            +hand_extra_weight * (left_hand_loss + right_hand_loss)
            loss.backward()
            print('stage 1, iter', i, 'mse loss =', loss.item(), end='\r')
            return loss

        optimizer.step(closure)

    #############################################
    # stage 2: fit everything
    #############################################
    optimizer = torch.optim.LBFGS([
        global_orient, transl, body_pose, betas, left_hand_pose,
        right_hand_pose, jaw_pose, expression
    ],
                                  max_iter=20,
                                  lr=1.0,
                                  line_search_fn='strong_wolfe')

    global_orient.requires_grad = True
    transl.requires_grad = True
    body_pose.requires_grad = True
    betas.requires_grad = True
    left_hand_pose.requires_grad = True
    right_hand_pose.requires_grad = True
    jaw_pose.requires_grad = True
    expression.requires_grad = True

    max_iter = 100
    face_extra_weight = 0.0
    hand_extra_weight = 0.0
    for i in range(max_iter):

        def closure():
            optimizer.zero_grad()

            output = body_model(
                global_orient=global_orient,
                transl=transl,
                body_pose=body_pose,
                betas=betas,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=jaw_pose,
                expression=expression,
                return_verts=True)
            pred_verts = output['vertices']
            face_loss = mse_loss(pred_verts[:, face_vertex_ids],
                                 target_verts[:, face_vertex_ids])
            left_hand_loss = l1_loss(pred_verts[:, left_hand_vertex_ids],
                                     target_verts[:, left_hand_vertex_ids])
            right_hand_loss = l1_loss(pred_verts[:, right_hand_vertex_ids],
                                      target_verts[:, right_hand_vertex_ids])
            all_loss = mse_loss(pred_verts, target_verts)
            loss = all_loss + face_extra_weight * face_loss
            +hand_extra_weight * (left_hand_loss + right_hand_loss)
            loss.backward()
            print('stage 2, iter', i, 'mse loss =', loss.item(), end='\r')
            return loss

        optimizer.step(closure)

    output = body_model(
        global_orient=global_orient,
        transl=transl,
        body_pose=body_pose,
        betas=betas,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        jaw_pose=jaw_pose,
        expression=expression,
        return_verts=True,
        return_joints=True,
    )
    pred_verts = output['vertices']
    pred_joints = output['joints']

    params = {
        'global_orient': global_orient.detach().cpu().numpy(),
        'transl': transl.detach().cpu().numpy(),
        'body_pose': body_pose.detach().cpu().numpy(),
        'betas': betas.detach().cpu().numpy(),
        'left_hand_pose': left_hand_pose.detach().cpu().numpy(),
        'right_hand_pose': right_hand_pose.detach().cpu().numpy(),
        'jaw_pose': jaw_pose.detach().cpu().numpy(),
        'expression': expression.detach().cpu().numpy(),
        'joints': pred_joints.detach().cpu().numpy(),
        'verts': pred_verts.detach().cpu().numpy(),
    }

    return params


def main(args):
    """This function fits a SMPL-X model to a given SMPL-X mesh.

    The result will be saved in same directory struct as the input mesh.
    """
    # parse data
    load_dir = args.load_dir
    mesh_type = args.mesh_type
    device = args.device
    basedir = osp.basename(load_dir)

    # prepare mesh type
    SUPPORTED_MESH_TYPES = ['obj', 'npy', 'ply']
    assert mesh_type in SUPPORTED_MESH_TYPES, \
        f'mesh type {mesh_type} not supported'

    # prepare body model
    smplx_model = build_body_model(
        dict(
            type='SMPLX',
            keypoint_src='smplx',
            keypoint_dst='smplx',
            model_path='data/body_models/smplx',
            gender='neutral',
            num_betas=10,
            use_face_contour=True,
            flat_hand_mean=False,
            use_pca=False,
            batch_size=1)).to(device)
    # faces = smplx_model.faces

    face_vids_path = 'data/body_models/smplx/SMPL-X__FLAME_vertex_ids.npy'
    args.face_vertex_ids = np.load(face_vids_path).astype(np.int32)
    hand_vertex_ids_path = 'data/body_models/smplx/MANO_SMPLX_vertex_ids.pkl'
    with open(hand_vertex_ids_path, 'rb') as f:
        vertex_idxs_data = pickle.load(f, encoding='latin1')
    args.left_hand_vertex_ids = vertex_idxs_data['left_hand']
    args.right_hand_vertex_ids = vertex_idxs_data['right_hand']

    # prepare file paths
    if args.recursive_search == 'True':
        file_ps = glob.glob(
            osp.join(load_dir, '**', f'*.{mesh_type}'), recursive=True)
    else:
        file_ps = glob.glob(osp.join(load_dir, f'*.{mesh_type}'))

    for fp in tqdm(
            file_ps,
            desc='fitting meshes',
            total=len(file_ps),
            position=0,
            leave=False):

        # prepare load pathnames
        if mesh_type == 'obj':
            target_verts = trimesh.load(fp).vertices.reshape(1, 10475, 3)
        if mesh_type == 'npy':
            target_verts = np.load(fp).reshape(1, 10475, 3)
        if mesh_type == 'ply':
            target_verts = trimesh.load(fp).vertices.reshape(1, 10475, 3)

        # fit parameters
        params = fit_params(target_verts, body_model=smplx_model, args=args)
        stem, _ = osp.splitext(osp.basename(fp))
        save_path = osp.join(load_dir, 'fitted_params', osp.basename(fp)).replace(f'.{args.mesh_type}', '.npz')
        os.makedirs(osp.dirname(save_path), exist_ok=True)

        # pdb.set_trace()
        np.savez(save_path, **params)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--load_dir',
        type=str,
        required=False,
        help='Directory to load data from',
        default='/mnt/d/sgnify/smplx_gt')
    parser.add_argument(
        '--mesh_type',
        type=str,
        required=False,
        help='Type of mesh file to fit',
        default='obj')

    # optional arguments
    parser.add_argument(
        '--recursive_search',
        type=str,
        required=False,
        help='Whether to recursively search for glob files',
        default='True')

    args = parser.parse_args()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    main(args)
