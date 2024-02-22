import numpy as np
import torch
import os
import pickle
import glob
import json
import tqdm
import pdb
import cv2
import argparse

from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model

from mmhuman3d.core.cameras import build_cameras

def main(args):

    # find all humandata
    hd_ps = glob.glob(args.hd_pp)
    dataset_path = (args.hd_pp).split('/output')[0]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for hd_p in hd_ps:
        # load humandata
        hd_param = dict(np.load(hd_p, allow_pickle=True))

        # get unqiue sequences
        seqs = list(set(hd_param['meta'].item()['sequence_name']))

        # get smplx and smpl anno type
        has_smpl, has_smplx = False, False
        bm = None
        if 'smplx' in hd_param.keys():
            has_smplx = True
            bm = 'smplx'
        elif 'smpl' in hd_param.keys():
            has_smpl = True
            bm = 'smpl'

        gendered_model = {}
        if has_smplx:
            for gender in ['male', 'female', 'neutral']:
                gendered_model[gender] = build_body_model(
                    dict(
                        type='SMPLX',
                        keypoint_src='smplx',
                        keypoint_dst='smplx',
                        model_path='data/body_models/smplx',
                        gender=gender,
                        num_betas=10,
                        use_face_contour=True,
                        flat_hand_mean=False,
                        use_pca=False,
                        batch_size=1)).to(device)
        elif has_smpl:
            for gender in ['male', 'female', 'neutral']:
                gendered_model[gender] = build_body_model(
                    dict(
                        type='SMPL',
                        keypoint_src='smpl_45',
                        keypoint_dst='smpl_45',
                        model_path='data/body_models/smpl',
                        gender=gender,
                        num_betas=10,
                        use_pca=False,
                        batch_size=1)).to(device)

        # for each sequence porject to 2d
        for seq in seqs:

            # get index
            idxs = [i for i, s in enumerate(hd_param['meta'].item()['sequence_name']) if s == seq]

            # load camera from frame 0
            image = cv2.imread(dataset_path + '/' + hd_param['image_path'][idxs[0]])
            focal_length = hd_param['meta'].item()['focal_length'][idxs[0]]
            camera_center = hd_param['meta'].item()['principal_point'][idxs[0]]
            if isinstance(focal_length, float):
                focal_length = [focal_length, focal_length]
            if isinstance(camera_center, float):
                camera_center = [camera_center, camera_center]

            camera_opencv = build_cameras(
                dict(type='PerspectiveCameras',
                    convention='opencv',
                    in_ndc=False,
                    focal_length=np.array([focal_length[0], focal_length[1]]).reshape(-1, 2),
                    principal_point=np.array([camera_center[0], camera_center[1]]).reshape(-1, 2),
                    image_size=(image.shape[0], image.shape[1]))).to(device)
            
            # load smplx params
            body_params = {}
            for body_key in hd_param[bm].item().keys():
                body_params[body_key] = hd_param[bm].item()[body_key][np.array(idxs)]
            
            body_params_tensor = {}
            for k, v in body_params.items():
                body_params_tensor[k] = torch.tensor(v).to(device)

            if 'gender' in hd_param['meta'].item().keys():
                gender = hd_param['meta'].item()['gender'][idxs[0]]
            else:
                gender = 'neutral'
                    
            # get smpl / smplx vertices
            output = gendered_model[gender](**body_params_tensor, return_verts=True)

            vertices = output['vertices'].detach().cpu().numpy()
            kps3d_c = output['joints'].detach().cpu().numpy()

            vertices_world = []
            kps3ds_w = []
            lowest_n = []

            # how many vertices for each frame
            vertices_threshold = 200

            for iid, idx in enumerate(idxs):

                # read RT
                RT = hd_param['meta'].item()['RT'][idx]
                RT_inv = np.linalg.inv(RT)

                # project vertices to world camera
                # vertice_world = np.matmul(RT_inv[:3, :3], vertices[iid].T) + RT_inv[:3, 3].reshape(-1, 1)
                # vertice_world = vertice_world.T
                # vertices_world.append(vertice_world)
                # lowest_n.append(np.sort(vertice_world[:, 2])[vertices_threshold])

                kps3d_w = np.matmul(RT_inv[:3, :3], kps3d_c[iid].T) + RT_inv[:3, 3].reshape(-1, 1)
                kps3d_w = kps3d_w.T
                kps3ds_w.append(kps3d_w)
                
            # vertices_world = np.array(vertices_world)
            # lowest_n.sort()



            # get lowest 100 point from 10% of the frames

            
            pdb.set_trace()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hd_pp', type=str, 
                      default='/mnt/d/datasets/sminchisescu-research-datasets/output/CHI3D*0605*')
    args = parser.parse_args()
    main(args)