import numpy as np
import glob
import random
import cv2
import os
import argparse
import torch
import pyrender
import trimesh
import pandas as pd

from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx, get_keypoint_idxs_by_part
from tools.convert_datasets import DATASET_CONFIGS
from mmhuman3d.models.body_models.builder import build_body_model
from tools.utils.request_files_server import request_files, request_files_name


from tools.vis.visualize_humandata import (render_pose, get_cam_params,
    dataset_path_dict, humandata_datasets,anno_path,camera_params_dict,
    smpl_shape, smplx_shape)

import pdb

# dataset status
local_datasets = ['gta_human2', 'egobody', 'ubody', 'ssp3d', 
                        'FIT3D', 'CHI3D', 'HumanSC3D']
# some datasets are on 1988
server_datasets = ['agora', 'arctic', 'bedlam', 'crowdpose', 'lspet', 'ochuman',
                    'posetrack', 'instavariety', 'mpi_inf_3dhp',
                    'mtp', 'muco3dhp', 'prox', 'renbody', 'rich', 'spec',
                    'synbody','talkshow', 'up3d', 'renbody_highres']

def sort_by_transl(npz_info):

    npz_info_dict = {}
    for [npzp, imgp] in npz_info:
        if not imgp in npz_info_dict.keys():
            npz_info_dict[imgp] = []
        npz_info_dict[imgp].append(npzp)
    
    # load transl
    for key in npz_info_dict.keys():
        npzps = npz_info_dict[key]
        distances = []
        for npzp in npzps:
            npz = dict(np.load(npzp))
            transl = npz['transl']
            # calculate distance
            distance = np.linalg.norm(transl)
            distances.append(distance)
        # pdb.set_trace()
        npzps = [x for _, x in sorted(zip(distances, npzps), reverse=True)]
        npz_info_dict[key] = npzps

    npz_info = []
    for key in npz_info_dict.keys():
        npz_info += [[npzp, key] for npzp in npz_info_dict[key]]

    return npz_info

def visualize_gt_eva(args):

    dataset_path = dataset_path_dict[args.dataset_name]
    dataset_name = args.dataset_name


    bmethod =  args.bmethod
    npzps = glob.glob(os.path.join(args.inputf, '**', '*.npz'), recursive=True)
    npzps = [p for p in npzps if dataset_name in p]
    npzps = [p for p in npzps if bmethod in p]
    random.shuffle(npzps)

    npz_info = []
    for npzp in tqdm(npzps, desc='reading npzs'):
        npz = dict(np.load(npzp))
        if args.dataset_name == 'agora':
            imgp = npz['img_path'].tolist()
            imgp = imgp.replace('/dataset/AGORA/data/', '')
            imgp = imgp.replace('_crop', '')
            imgp = imgp.split('_ann_id')[0] + imgp[-4:]
            selected_images = ['ag_validationset_renderpeople_bfh_hdri_50mm_5_15_00103.png',
                                'ag_validationset_renderpeople_bfh_hdri_50mm_5_15_00063.png',
                                'ag_validationset_renderpeople_bfh_hdri_50mm_5_15_00061.png',
                                'ag_validationset_renderpeople_bfh_hdri_50mm_5_15_00017.png',
                                'ag_validationset_renderpeople_bfh_hdri_50mm_5_15_00011.png',
                                'ag_validationset_renderpeople_bfh_flowers_5_15_00108.png']
            # pdb.set_trace()
            if not os.path.basename(imgp) in selected_images:
                continue
        if args.dataset_name == 'ehf':
            imgp = os.path.basename(npzp).replace('npz', 'png')
        if args.dataset_name == 'egobody':
            imgp = npz['img_path'].tolist()
            imgp = imgp.replace('/dataset/EgoBody/', '')
            # pdb.set_trace()
        if args.dataset_name == 'ubody':
            pdb.set_trace()
        if args.dataset_name == 'arctic':
            imgp = npz['img_path'].tolist()
            imgp = imgp.replace('/dataset/ARCTIC/', '')
 
        # get image path
        npz_info.append([npzp, imgp])
        # npz_info = npz_info[:1000]
    # pdb.set_trace()
    # use humandata to load index
    if dataset_name not in server_datasets:
        param_ps = glob.glob(os.path.join(dataset_path_dict[dataset_name], 
                                            anno_path[dataset_name], f'*{dataset_name}*.npz'))
        # param_ps = [p for p in param_ps if 'test' not in p]s
        # param_ps = [p for p in param_ps if 'val' not in p]
    else:
        data_info = [info for info in humandata_datasets if info[0] == dataset_name][0]
        _, filename, exclude = data_info

        param_ps = glob.glob(os.path.join(args.server_local_path, filename))
        if exclude != '':
            param_ps = [p for p in param_ps if exclude not in p]

    # pdb.set_trace()

    for npz_id, param_p in enumerate(param_ps[:1]):
        # pdb.set_trace()
        param = dict(np.load(param_p, allow_pickle=True))
        # pdb.set_trace()
        # ----------temporal align for single person dataset----------
        # anno_indexs = []
        # for idx, imgp in enumerate(tqdm(param['image_path'], 
        #                       desc=f'aligning npz and anno {npz_id+1}/{len(param_ps)}', 
        #                         position=0, leave=False)): 
        #     for [npzp, image_path] in npz_info:
        #         if image_path == imgp:
        #             anno_index = idx
        #             if anno_index in anno_indexs:
        #                 continue
        #             anno_indexs.append([npzp, idx])

        # B_d = {image_path: npzp for [npzp, image_path] in npz_info}
        # anno_indexs = []
        # for idx, imgp in enumerate(tqdm(param['image_path'],
        #                         desc=f'aligning npz and anno {npz_id+1}/{len(param_ps)}', 
        #                         position=0, leave=False)):
        #     if imgp in B_d:
        #         anno_indexs.append([B_d[imgp], idx])
        # print(f'Sort {os.path.basename(param_p)} finished, found {len(anno_indexs)} matches')
        # pdb.set_trace()
        # if len(anno_indexs) == 0:
        #     continue

        # check for params
        has_smplx, has_smpl, has_gender = False, False, False
        if 'smplx' in param.keys():
            has_smplx = True
        elif 'smpl' in param.keys():
            has_smpl = True

        # check for params
        has_smplx, has_smpl, has_gender = False, False, False
        if 'smplx' in param.keys():
            has_smplx = True
        elif 'smpl' in param.keys():
            has_smpl = True


        # load params
        if has_smpl:
            body_model_param_smpl = param['smpl'].item()
        if has_smplx:
            body_model_param_smplx = param['smplx'].item()  
            if dataset_name == 'bedlam':
                body_model_param_smplx['betas'] = body_model_param_smplx['betas'][:, :10]
                flat_hand_mean = True

        if 'meta' in param.keys():
            if 'gender' in param['meta'].item().keys():
                has_gender = True
        # read smplx only if has both smpl and smplx
        if has_smpl and has_smplx:
            has_smpl = False
        has_smplx = True
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # build smpl model
        if has_smpl:
            smpl_model = {}
            for gender in ['male', 'female', 'neutral']:
                smpl_model[gender] = build_body_model(dict(
                        type='SMPL',
                        keypoint_src='smpl_45',
                        keypoint_dst='smpl_45',
                        model_path='data/body_models/smpl',
                        gender=gender,
                        num_betas=10,
                        use_face_contour=True,
                        flat_hand_mean=flat_hand_mean,
                        use_pca=False,
                        batch_size=1
                    )).to(device)
                
        # build smplx model
        if has_smplx:
            smplx_model = {}
            for gender in ['male', 'female', 'neutral']:
                smplx_model[gender] = build_body_model(dict(
                        type='SMPLX',
                        keypoint_src='smplx',
                        keypoint_dst='smplx',
                        model_path='data/body_models/smplx',
                        gender=gender,
                        num_betas=10,
                        use_face_contour=True,
                        flat_hand_mean=False,
                        use_pca=False,
                        batch_size=1
                    )).to(device)
            smplx_model_fhm = {}
            for gender in ['male', 'female', 'neutral']:
                smplx_model_fhm[gender] = build_body_model(dict(
                        type='SMPLX',
                        keypoint_src='smplx',
                        keypoint_dst='smplx',
                        model_path='data/body_models/smplx',
                        gender=gender,
                        num_betas=10,
                        use_face_contour=True,
                        flat_hand_mean=True,
                        use_pca=False,
                        batch_size=1
                    )).to(device)

        # pdb.set_trace()
        # prepare for server request if datasets on server
        if dataset_name in server_datasets:
            # print('getting images from server...')
            # if isinstance(param['image_path'], np.ndarray):
            #     param['image_path'] = param['image_path'].tolist()
            # files = [param['image_path'][fid] for [_, fid ]in anno_indexs]
            if dataset_name == 'agora':
                files = [f for [_, f] in npz_info]
            # pdb.set_trace()
            local_image_folder = os.path.join(args.image_cache_path, dataset_name)
            
            # check for existing images
            files = [f for f in files if not os.path.exists(os.path.join(local_image_folder, f))]

            os.makedirs(local_image_folder, exist_ok=True)
            files = list(set(files))
            # pdb.set_trace()
            request_files(files, 
                        server_path=dataset_path_dict[dataset_name], 
                        local_path=local_image_folder, 
                        server_name='1988')
            # print('done')
        else:
            local_image_folder = dataset_path_dict[dataset_name]

    # for idx in tqdm(sorted(idxs), desc=f'Processing npzs {npz_id}/{len(param_ps)}, sample size: {sample_size}',
    #                 position=0, leave=False):

        print(f'Saving image to {os.path.join(args.out_path, f"{dataset_name}_{bmethod}")}')
        # for idx, [npzp, imgp] in enumerate(tqdm(npz_info, desc=f'Processing npzs {npz_id+1}/{len(param_ps)}',
        #                 position=0, leave=False)):
        imgpsu = list(set([f for [_, f] in npz_info]))

        npz_info = sort_by_transl(npz_info)

        image_dict = {}
        for idx, [npzp, imgp] in enumerate(npz_info):
            anno_param_dict = dict(np.load(npzp, allow_pickle=True))
            idx = imgpsu.index(imgp)
            print(idx)
            # image_p = param['image_path'][anno_idx]
            image_p = imgp
            image_path = os.path.join(local_image_folder, image_p)
            # pdb.set_trace()
            if imgp not in image_dict.keys():
                image = cv2.imread(image_path)

            else:
                # pdb.set_trace()
                image = image_dict[imgp]
            # convert to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # temporal fix for betas and gender
            if dataset_name in ['bedlam']:
                has_gender = False

            # ---------------------- render single pose ------------------------
            # read cam params
            # focal_length, camera_center, R, T = get_cam_params(
            #                 camera_params_dict, args.dataset_name, param, idx)
            R, T = None, None
            # read gender
            # if has_gender:
            #     try:
            #         gender = param['meta'].item()['gender'][1]
            #         if gender == 'f':
            #             gender = 'female'
            #         elif gender == 'm':
            #             gender = 'male'
            #     except IndexError: 
            #         gender = 'neutral'
            # else:
            #     gender = 'neutral'
            gender = 'neutral'

            # prepare for mesh projection
            if dataset_name == 'agora':
                if 'flowers' in image_p:
                    focal_length, camera_center = [2986.66666667, 2986.66666667], [1920., 1080.]
                elif 'hdri' in image_p:
                    focal_length, camera_center = [5333.33333333, 5333.33333333], [1920., 1080.]
            camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0], fy=focal_length[1],
                cx=camera_center[0], cy=camera_center[1])
            
            # ---------------------- render eva results ----------------------
            # use gender = neutral for eva
            if has_smpl:
                intersect_key = list(set(anno_param_dict.keys()) & set(smpl_shape.keys()))
                body_model_param_tensor = {key: torch.tensor(
                        np.array(anno_param_dict[key]).reshape(smpl_shape[key]),
                                device=device, dtype=torch.float32)
                                for key in intersect_key
                                if len(anno_param_dict[key]) > 0}
                # use transl in gt
                # body_model_param_tensor['transl'] = torch.tensor(
                #         np.array(body_model_param_smpl['transl'][idx:idx+1]).reshape(smpl_shape['transl']),
                #         device=device, dtype=torch.float32)
                
                # temporal fix for ochuman, lspet, posetrack
                if dataset_name in ['ochuman', 'lspet', 'posetrack']:
                    zfar, znear= 600, 30
                    camera = pyrender.camera.IntrinsicsCamera(
                        fx=focal_length[0], fy=focal_length[1],
                        cx=camera_center[0], cy=camera_center[1],
                        zfar=zfar, znear=znear)
                    rendered_image = render_pose(img=image, 
                                            body_model_param=body_model_param_tensor, 
                                            body_model=smpl_model['neutral'],
                                            camera=camera,
                                            dataset_name=dataset_name,
                                            R=R, T=T)             
                else:
                    rendered_image = render_pose(img=image, 
                                        body_model_param=body_model_param_tensor, 
                                        body_model=smpl_model['neutral'],
                                        camera=camera,
                                        dataset_name=dataset_name,
                                        R=R, T=T)

            if has_smplx:  
                # pdb.set_trace()
                intersect_key = list(set(anno_param_dict.keys()) & set(smplx_shape.keys()))
                body_model_param_tensor = {key: torch.tensor(
                        np.array(anno_param_dict[key]).reshape(smplx_shape[key]),
                                device=device, dtype=torch.float32)
                                for key in intersect_key
                                if len(anno_param_dict[key]) > 0}
                # use transl in gt
                # body_model_param_tensor['transl'] = torch.tensor(
                #         np.array(param['smplx'].item()['transl'][anno_idx]).reshape(smplx_shape['transl']),
                #         device=device, dtype=torch.float32)
                rendered_image = render_pose(img=image, 
                                            body_model_param=body_model_param_tensor, 
                                            body_model=smplx_model['neutral'],
                                            camera=camera,
                                            dataset_name=dataset_name,
                                            R=R, T=T, camera_opencv=None)
                image_dict[imgp] = rendered_image
        
        # ---------------------- save eva render results ----------------------
        for key in image_dict.keys():
            print(key, image_dict[key].shape)
            rendered_image = image_dict[key]
            os.makedirs(os.path.join(args.out_path, f'{dataset_name}_{bmethod}'), exist_ok=True)
            out_image_path = os.path.join(args.out_path, f'{dataset_name}_{bmethod}',
                        f'{os.path.basename(key)}.png')
            cv2.imwrite(out_image_path, rendered_image)

    # pdb.set_trace()



if __name__ == '__main__':

    # main args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='agora')
    parser.add_argument('--bmethod', type=str, default='osx')
    parser.add_argument('--inputf', type=str, 
                        default='/mnt/c/users/12595/desktop/aios-vis/input')
    parser.add_argument('--out_path', type=str,
                        default='/mnt/c/users/12595/desktop/aios-vis/output')

    # optional args
    parser.add_argument('--img_base_path', type=str, required=False,
                        default='output')
    parser.add_argument('--cache_path', type=str, default='/mnt/d/image_cache_1988',
                        required=False)
    args = parser.parse_args()

    args.server_local_path = '/mnt/d/annotations_1988'
    args.image_cache_path = args.cache_path
    args.flat_hand_mean = False

    visualize_gt_eva(args)