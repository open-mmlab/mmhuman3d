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
from tools.vis.visualize_humandata_qp import render_pose as render_pose_qp
import pdb

smpl_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 69)}
smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}
smplx_shape_except_expression = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3)}
smplx_shape = smplx_shape_except_expression

def get_cam_params(camera_params_dict, dataset_name, param, idx):

    eft_datasets = ['ochuman','lspet', 'posetrack']

    R, T = None, None
    # read cam params
    if dataset_name in camera_params_dict.keys():
        cx, cy = camera_params_dict[dataset_name]['principal_point']
        fx, fy = camera_params_dict[dataset_name]['focal_length']
        camera_center = (cx, cy)
        focal_length = (fx, fy)
        if R in camera_params_dict[dataset_name].keys():
            R = camera_params_dict[dataset_name]['R']
        if T in camera_params_dict[dataset_name].keys():
            T = camera_params_dict[dataset_name]['T']
    elif dataset_name in eft_datasets:
        pred_cam = param['meta'].item()['pred_cam'][idx]
        cam_scale, cam_trans = pred_cam[0], pred_cam[1:]
        # pdb.set_trace()
        bbox_xywh = param['bbox_xywh_vis'][idx][:4]
        R = np.eye(3) * cam_scale * bbox_xywh[-1] / 2
        T = np.array([
                (cam_trans[0]+1)*bbox_xywh[-1]/2 + bbox_xywh[0], 
                (cam_trans[1]+1)*bbox_xywh[-1]/2 + bbox_xywh[1], 
                0])
        focal_length = [5000, 5000]
        camera_center = [0, 0]
    else:
        try:
            focal_length = param['meta'].item()['focal_length'][idx]
            camera_center = param['meta'].item()['principal_point'][idx]
        except KeyError:
            focal_length = param['misc'].item()['focal_length']
            camera_center = param['meta'].item()['principal_point']
        except TypeError:
            focal_length = param['meta'].item()['focal_length']
            camera_center = param['meta'].item()['princpt']
        try:
            R = param['meta'].item()['R'][idx]
            T = param['meta'].item()['T'][idx]
        except KeyError:
            R = None
            T = None
        except IndexError:
            R = None
            T = None
        
            
    focal_length = np.asarray(focal_length).reshape(-1)
    camera_center = np.asarray(camera_center).reshape(-1)

    if len(focal_length)==1:
        focal_length = [focal_length, focal_length]
    if len(camera_center)==1:
        camera_center = [camera_center, camera_center]

    return focal_length, camera_center, R, T


def render_pose(img, body_model_param, body_model, camera, return_mask=False,
                 R=None, T=None, dataset_name=None, camera_opencv=None):

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    output = body_model(**body_model_param, return_verts=True)
    
    vertices = output['vertices'].detach().cpu().numpy().squeeze()

    # # overlay on img
    # verts = output['vertices']
    # verts2d = camera_opencv.transform_points_screen(verts)[..., :2].detach().cpu().numpy()
    
    # for i in range(verts2d.shape[1]):
    #     cv2.circle(img, (int(verts2d[0, i, 0]), int(verts2d[0, i, 1])), 1, (0, 255, 0), -1)


    eft_datasets = ['ochuman', 'lspet', 'posetrack']
    # adjust vertices beased on R and T
    if R is not None:
        joints = output['joints'].detach().cpu().numpy().squeeze()
        root_joints = joints[0]
        if dataset_name in eft_datasets:
            vertices = np.dot(R, vertices.transpose(1,0)).transpose(1,0) + T.reshape(1,3)
            vertices[:, 2] = (vertices[:, 2]-vertices[:, 2].min())/ (vertices[:, 2].max()-vertices[:, 2].min())*30 +500
            vertices[:, :2] *= vertices[:, [2]]
            vertices[:, :2] /= 5000.
        else:
            T = np.dot(np.array(R), root_joints) - root_joints  + np.array(T)
            vertices = vertices + T

    faces = body_model.faces

    # render material
    base_color = (1.0, 193/255, 193/255, 1.0)
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=base_color)
    
    # get body mesh
    body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    # prepare camera and light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_pose = pyrender2opencv @ np.eye(4)
    
    # build scene
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=cam_pose)
    scene.add(body_mesh, 'mesh')

    # render scene
    os.environ["PYOPENGL_PLATFORM"] = "osmesa" # include this line if use in vscode
    r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                    viewport_height=img.shape[0],
                                    point_size=1.0)
    
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    # alpha = 1.0  # set transparency in [0.0, 1.0]
    # color[:, :, -1] = color[:, :, -1] * alpha
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    img = img / 255
    # output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * img)
    opacity = 0.8
    output_img = (color[:, :, :] * valid_mask * opacity + (1 - valid_mask) * img +
                   valid_mask * (1 - opacity) * img)

    # output_img = color

    img = (output_img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8)

    return img


def visualize_gt_eva(args, image_idxs, image_paths, anno_param_dicts, ranked=True):

    # dataset status
    local_datasets = ['gta_human2', 'egobody', 'ubody', 'ssp3d', 
                          'FIT3D', 'CHI3D', 'HumanSC3D']
    # some datasets are on 1988
    server_datasets = ['agora', 'arctic', 'bedlam', 'crowdpose', 'lspet', 'ochuman',
                       'posetrack', 'instavariety', 'mpi_inf_3dhp',
                       'mtp', 'muco3dhp', 'prox', 'renbody', 'rich', 'spec',
                       'synbody','talkshow', 'up3d', 'renbody_highres']
    
    humandata_datasets = [ # name, glob pattern, exclude pattern
        ('agora', 'agora*forvis*.npz', ''),
        ('arctic', 'p1_*.npz', ''),
        ('bedlam', 'bedlam_train.npz', ''),
        ('behave', 'behave_train_230516_231_downsampled.npz', ''),
        ('chi3d', 'CHI3D_train_230511_1492_*.npz', ''),
        ('crowdpose', 'crowdpose_neural_annot_train_new.npz', ''),
        ('lspet', 'eft_lspet*.npz', ''),
        ('ochuman', 'eft_ochuman*.npz', ''),
        ('posetrack', 'eft_posetrack*.npz', ''),
        ('egobody_ego', 'egobody_egocentric_train_230425_065_fix_betas.npz', ''),
        ('egobody_kinect', 'egobody_kinect_train_230503_065_fix_betas.npz', ''),
        ('fit3d', 'FIT3D_train_230511_1504_*.npz', ''),
        ('gta', 'gta_human2multiple_230406_04000_0.npz', ''),
        ('ehf', 'h4w_ehf_val_updated_v2.npz', ''),  # use humandata ehf to get bbox
        ('humansc3d', 'HumanSC3D_train_230511_2752_*.npz', ''),
        ('instavariety', 'insta_variety_neural_annot_train.npz', ''),
        ('mpi_inf_3dhp', 'mpi_inf_3dhp_neural_annot_train.npz', ''),
        ('mtp', 'mtp_smplx_train.npz', ''),
        ('muco3dhp', 'muco3dhp_train.npz', ''),
        ('prox', 'prox_train_smplx_new.npz', ''),
        ('renbody', 'renbody_train_230525_399_*.npz', ''),
        ('renbody_highres', 'renbody*highrescam*_fix_betas.npz', ''),
        ('rich', 'rich_train_fix_betas.npz', ''),
        ('spec', 'spec_train_smpl.npz', ''),
        ('ssp3d', 'ssp3d_230525_311.npz', ''),
        ('synbody_magic1', 'synbody_amass_230328_02172.npz', ''),
        ('synbody', 'synbody_train_230521_04000_fix_betas.npz', ''),
        ('talkshow', 'talkshow_smplx_*.npz', 'path'),
        ('up3d', 'up3d*.npz', ''),
    ]

    dataset_path_dict = {
        'agora': '/lustrenew/share_data/caizhongang/data/datasets/agora',
        'arctic': '/lustre/share_data/weichen1/arctic/unpack/arctic_data/data/images',
        'behave': '/mnt/e/behave',
        'bedlam': '/lustre/share_data/weichen1/bedlam/train_images',
        'CHI3D': '/mnt/d/sminchisescu-research-datasets',
        'crowdpose': '/lustrenew/share_data/zoetrope/data/datasets/crowdpose',
        'lspet': '/lustrenew/share_data/zoetrope/data/datasets/hr-lspet',
        'ochuman': '/lustrenew/share_data/zoetrope/data/datasets/ochuman',
        'egobody': '/mnt/d/egobody',
        'FIT3D': '/mnt/d/sminchisescu-research-datasets',
        'gta_human2': '/mnt/e/gta_human2',
        'ehf': '/mnt/e/ehf',
        'HumanSC3D': '/mnt/d/sminchisescu-research-datasets',
        'instavariety': '/lustrenew/share_data/zoetrope/data/datasets/neural_annot_data/insta_variety/',
        'mpi_inf_3dhp': '/lustrenew/share_data/zoetrope/osx/data/MPI_INF_3DHP_folder/data/',
        'mtp': '/lustre/share_data/weichen1/mtp',
        'muco3dhp': '/lustre/share_data/weichen1/MuCo',
        'posetrack': 'lustrenew/share_data/zoetrope/data/datasets/posetrack/data/images',
        'prox': '/lustre/share_data/weichen1/PROXFlip',
        'renbody': '/lustre/share_data/weichen1/renbody',
        'renbody_highres': '/lustre/share_data/weichen1/renbody',
        'rich': '/lustrenew/share_data/zoetrope/data/datasets/rich/images/train',
        'spec': '/lustre/share_data/weichen1/spec/',
        'ssp3d': '/mnt/e/ssp-3d',
        'synbody': '/lustre/share_data/meihaiyi/shared_data/SynBody',
        'synbody_magic1': '/lustre/share_data/weichen1/synbody',
        'talkshow': '/lustre/share_data/weichen1/talkshow_frames',
        'ubody': '/mnt/d/ubody',
        'up3d': '/lustrenew/share_data/zoetrope/data/datasets/up3d/up-3d/up-3d',
        }
    
    anno_path = {
        'shapy': 'output',
        'gta_human2': 'output',
        'egobody': 'output',
        'ubody': 'output/intra',
        'ssp3d': 'output',
        'FIT3D': 'output',
        'CHI3D': 'output',
        'HumanSC3D': 'output',
        'behave': 'output',
        'ehf': 'output'}
    
    camera_params_dict = {
        'gta_human2': {'principal_point': [960, 540], 'focal_length': [1158.0337, 1158.0337]},
        'ssp3d': {'focal_length': [5000, 5000], 'principal_point': [256, 256]},
        'synbody': {'focal_length': [640, 640], 'principal_point': [640, 360]},
        'ehf': {'focal_length': [1498.224, 1498.224], 'principal_point': [790.263706, 578.90334],
                'T': np.array([-0.03609917, 0.43416458, 2.37101226]),
                'R': np.array([[ 0.9992447 , -0.00488005,  0.03855169],
                            [-0.01071995, -0.98820424,  0.15276562],
                            [ 0.03735144, -0.15306349, -0.9875102 ]]),},
    }

    # load humandata
    dataset_name = args.dataset_name
    flat_hand_mean = args.flat_hand_mean

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

    # -------------------align image path-------------------
    # ehf
    if dataset_name == 'ehf':
        image_paths = [os.path.basename(imgp) for imgp in image_paths] 
    if dataset_name == 'arctic':
        image_paths = [os.path.sep.join(imgp.split(os.path.sep)[-4:]) for imgp in image_paths]
    if dataset_name == 'egobody':
        image_paths = [os.path.sep.join(imgp.split(os.path.sep)[3:]) for imgp in image_paths]
    if dataset_name == 'ubody':
        image_paths = [os.path.sep.join(imgp.split(os.path.sep)[3:]) for imgp in image_paths]
    if dataset_name == 'agora':
        image_paths = [os.path.sep.join(imgp.split(os.path.sep)[4:]) for imgp in image_paths]
        image_paths = [f'{imgp.split("_ann_id_")[0]}.png' for imgp in image_paths]
        image_paths = [imgp.replace('validation_crop', 'validation') for imgp in image_paths]
    if dataset_name in ['renbody', 'renbody_highres']:
        image_paths = [os.path.sep.join(imgp.split(os.path.sep)[3:]) for imgp in image_paths]
    # pdb.set_trace()
    # render
    # for npz_id, param_p in enumerate(tqdm(param_ps, desc=f'Processing npzs',
    #                     position=0, leave=False)):
    for npz_id, param_p in enumerate(tqdm(param_ps)):
        param = dict(np.load(param_p, allow_pickle=True))
        # pdb.set_trace()
        # ----------temporal align for single person dataset----------
        idxs, anno_indexs = [], []
        for idx, imgp in enumerate(param['image_path']): 
            if imgp in image_paths:
                anno_index = image_paths.index(imgp)
                idxs.append(idx)
                anno_indexs.append(anno_index)

        if len(idxs) == 0:
            continue

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

        # prepare for server request if datasets on server
        if dataset_name in server_datasets:
            # print('getting images from server...')
            files = param['image_path'][idxs]
            # pdb.set_trace()
            local_image_folder = os.path.join(args.image_cache_path, dataset_name)
            os.makedirs(local_image_folder, exist_ok=True)
            request_files(files, 
                        server_path=dataset_path_dict[dataset_name], 
                        local_path=local_image_folder, 
                        server_name='1988')
            # print('done')
        else:
            local_image_folder = dataset_path_dict[dataset_name]

        # for idx in tqdm(sorted(idxs), desc=f'Processing npzs {npz_id}/{len(param_ps)}, sample size: {sample_size}',
        #                 position=0, leave=False):
        for idx, anno_idx in zip(idxs, anno_indexs):
            anno_param_dict = anno_param_dicts[anno_idx]
            rank = anno_idx

            image_p = param['image_path'][idx]
            image_path = os.path.join(local_image_folder, image_p)
            image = cv2.imread(image_path)
            # convert to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # temporal fix for betas and gender
            if dataset_name in ['bedlam']:
                has_gender = False

            # ---------------------- render single pose ------------------------
            # read cam params
            focal_length, camera_center, R, T = get_cam_params(
                            camera_params_dict, args.dataset_name, param, idx)
    
            # read gender
            if has_gender:
                try:
                    gender = param['meta'].item()['gender'][idx]
                    if gender == 'f':
                        gender = 'female'
                    elif gender == 'm':
                        gender = 'male'
                except IndexError: 
                    gender = 'neutral'
            else:
                gender = 'neutral'

            # prepare for mesh projection
            camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0], fy=focal_length[1],
                cx=camera_center[0], cy=camera_center[1])

            bmarks = list(anno_param_dict.keys())
            for bmark in bmarks + ['gt']:

                if has_smpl:
                    if bmark == 'gt':
                        intersect_key = list(set(body_model_param_smpl.keys()) & set(smpl_shape.keys()))
                        body_model_param_tensor = {key: torch.tensor(
                                np.array(body_model_param_smpl[key][idx:idx+1]).reshape(smpl_shape[key]),
                                        device=device, dtype=torch.float32)
                                        for key in intersect_key
                                        if len(body_model_param_smpl[key][idx:idx+1]) > 0}
                    else:
                        intersect_key = list(set(anno_param_dict[bmark].keys()) & set(smpl_shape.keys()))
                        body_model_param_tensor = {key: torch.tensor(
                                np.array(anno_param_dict[bmark][key]).reshape(smpl_shape[key]),
                                        device=device, dtype=torch.float32)
                                        for key in intersect_key
                                        if len(anno_param_dict[bmark][key]) > 0}
                    # use transl in gt
                    body_model_param_tensor['transl'] = torch.tensor(
                            np.array(body_model_param_smpl['transl'][idx:idx+1]).reshape(smpl_shape['transl']),
                            device=device, dtype=torch.float32)
                    
                    # temporal fix for ochuman, lspet, posetrack
                    if dataset_name in ['ochuman', 'lspet', 'posetrack']:
                        zfar, znear= 600, 30
                        camera = pyrender.camera.IntrinsicsCamera(
                            fx=focal_length[0], fy=focal_length[1],
                            cx=camera_center[0], cy=camera_center[1],
                            zfar=zfar, znear=znear)
                        rendered_image = render_pose(img=image, 
                                                body_model_param=body_model_param_tensor, 
                                                body_model=smpl_model[gender],
                                                camera=camera,
                                                dataset_name=dataset_name,
                                                R=R, T=T)             
                    else:
                        rendered_image = render_pose(img=image, 
                                            body_model_param=body_model_param_tensor, 
                                            body_model=smpl_model[gender],
                                            camera=camera,
                                            dataset_name=dataset_name,
                                            R=R, T=T)

                if has_smplx:
                    if bmark == 'gt':
                        intersect_key = list(set(body_model_param_smplx.keys()) & set(smplx_shape.keys()))
                        body_model_param_tensor = {key: torch.tensor(
                                np.array(body_model_param_smplx[key][idx:idx+1]).reshape(smplx_shape[key]),
                                        device=device, dtype=torch.float32)
                                        for key in intersect_key
                                        if len(body_model_param_smplx[key][idx:idx+1]) > 0}
                        # arctic uses flat hand mean = True
                        if dataset_name in ['arctic']:
                            rendered_image = render_pose(img=image, 
                                                        body_model_param=body_model_param_tensor, 
                                                        body_model=smplx_model_fhm[gender],
                                                        camera=camera,
                                                        dataset_name=dataset_name,
                                                        R=R, T=T, camera_opencv=None)
                        else:
                            rendered_image = render_pose(img=image, 
                                                        body_model_param=body_model_param_tensor, 
                                                        body_model=smplx_model[gender],
                                                        camera=camera,
                                                        dataset_name=dataset_name,
                                                        R=R, T=T, camera_opencv=None)                           
                    else:
                        intersect_key = list(set(anno_param_dict[bmark].keys()) & set(smplx_shape.keys()))
                        body_model_param_tensor = {key: torch.tensor(
                                np.array(anno_param_dict[bmark][key]).reshape(smplx_shape[key]),
                                        device=device, dtype=torch.float32)
                                        for key in intersect_key
                                        if len(anno_param_dict[bmark][key]) > 0}
                        # use transl in gt
                        body_model_param_tensor['transl'] = torch.tensor(
                                np.array(body_model_param_smplx['transl'][idx:idx+1]).reshape(smplx_shape['transl']),
                                device=device, dtype=torch.float32)
                        
                        rendered_image = render_pose(img=image, 
                                                    body_model_param=body_model_param_tensor, 
                                                    body_model=smplx_model['neutral'],
                                                    camera=camera,
                                                    dataset_name=dataset_name,
                                                    R=R, T=T, camera_opencv=None)
                    
                # ---------------------- render results ----------------------
                # print(f'Truncation: {trunc}')
                os.makedirs(os.path.join(args.out_path, args.dataset_name), exist_ok=True)

                # for writting on image
                font_size = image.shape[1] / 1000
                line_sickness = int(image.shape[1] / 1000) + 1
                front_location_y = int(image.shape[1] / 10)
                front_location_x = int(image.shape[0] / 10)

                # write rank
                if ranked:
                    out_image_path = os.path.join(args.out_path, args.dataset_name,
                        f'{"{:03d}".format(rank)}_{os.path.basename(param_ps[npz_id])[:-4]}_{idx}_{bmark}.png')
                else:
                    out_image_path = os.path.join(args.out_path, args.dataset_name,
                                f'{os.path.basename(param_ps[npz_id])[:-4]}_{idx}_{bmark}.png')
                print(f'Saving image to {out_image_path}')
                cv2.imwrite(out_image_path, rendered_image)

            # write rank
            if ranked:
                out_image_path_org = os.path.join(args.out_path, args.dataset_name,
                    f'{"{:03d}".format(rank)}_{os.path.basename(param_ps[npz_id])[:-4]}_{idx}_zoriginal.png')
            else:
                out_image_path_org = os.path.join(args.out_path, args.dataset_name,
                            f'{os.path.basename(param_ps[npz_id])[:-4]}_{idx}_zoriginal.png')
            # convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(out_image_path_org, image)

                # pdb.set_trace()


def ftp_target_anno(anno_base_dir, anno_dir, anno_names, local_dir):

    # pdb.set_trace()
    os.makedirs(os.path.join(local_dir, anno_dir), exist_ok=True)

    # read local file
    pending_download = []
    for anno_name in anno_names:
        target_local_file = os.path.join(local_dir, anno_dir, anno_name)
        if not os.path.exists(target_local_file):
            pending_download.append(os.path.join(anno_dir, anno_name))
    if len(pending_download) > 0:
        request_files(pending_download, 
                    server_path=anno_base_dir, 
                    local_path=local_dir, 
                    server_name='1988') 
        

def visualize_eva(args):

    # dataset status
    eva_datasets = ['gta_human2', 'egobody', 'ubody', 'ssp3d', 
                          'FIT3D', 'CHI3D', 'HumanSC3D']
    # on 1988
    eva_output = '/lustrenew/share_data/zoetrope/osx/output_wanqi'

    eva_keys = ['H4W', 'OSX', 'L32', 'H20']
    # dict_path for every benchmark output
    eva_anno_dict = dict(
        ehf={
            'H4W': 'test_h4w_vis_ep7_EHF_20230606_180847/vis',
            'OSX': 'test_osx_vis_ep999_EHF_20230606_174828/vis',
            'L32': 'test_exp114_vis_ep3_EHF_20230607_164910/vis',
            'H20': 'test_exp117_vis_ep4_EHF_20230607_164825/vis',
        },
        agora={
            'H4W': 'test_h4w_vis_ep7_AGORA_val_20230607_145313/vis',
            'OSX': 'test_osx_vis_ep999_AGORA_val_20230607_144123/vis',
            'L32': 'test_exp114_vis_ep3_AGORA_val_20230607_151156/vis',
            'H20': 'test_exp117_vis_ep4_AGORA_val_20230607_160510/vis',
        },
        ubody={
            'H4W': 'test_h4w_vis_ep7_UBody_20230607_150602/vis',
            'OSX': 'test_osx_vis_ep999_UBody_20230607_145041/vis',
            'L32': 'test_exp114_vis_ep3_UBody_20230607_153803/vis',
            'H20': 'test_exp117_vis_ep4_UBody_20230607_153854/vis',
        },
        egobody={
            'H4W': 'test_h4w_vis_ep7_EgoBody_Egocentric_20230607_153925/vis',
            'OSX': 'test_osx_vis_ep999_EgoBody_Egocentric_20230607_152442/vis',
            'L32': 'test_exp114_vis_ep3_EgoBody_Egocentric_20230607_163538/vis',
            'H20': 'test_exp117_vis_ep4_EgoBody_Egocentric_20230607_163546/vis',
        },
        arctic={
            'H4W': 'test_h4w_vis_ep7_ARCTIC_20230607_184338/vis',
            'OSX': 'test_osx_vis_ep999_ARCTIC_20230607_182625/vis',
            'L32': 'test_exp114_vis_ep3_ARCTIC_20230607_203528/vis',
            'H20': 'test_exp117_vis_ep4_ARCTIC_20230607_204153/vis',
        },
        renbody_highres={
            'H4W': 'test_h4w_vis_ep7_RenBody_20230607_213652/vis',
            'OSX': 'test_osx_vis_ep999_RenBody_20230607_200129/vis',
            'L32': 'test_exp114_vis_ep3_RenBody_20230607_214327/vis',
            'H20': 'test_exp117_vis_ep4_RenBody_20230607_212645/vis',
        }
    )
    # selected frame
    eva_selection_path = '/home/weichen/zoehuman/mmhuman3d/tools/vis_neurlips'
    eva_selection_dict = dict(
        ehf='select_ehf_smplx_error.csv',
        agora='select_agora_smplx_error.csv',
        ubody='select_UBody_smplx_error.csv',
        egobody='select_EgoBody_Egocentric_smplx_error.csv',
        arctic='select_ARCTIC_smplx_error.csv',
        renbody_highres='select_RenBody_HiRes_smplx_error.csv',
    )

    # prepare visualize input
    image_idxs, image_paths = [], []
    anno_param_dicts = []

    selected_frame_df = pd.read_csv(os.path.join(
            eva_selection_path, eva_selection_dict[args.dataset_name]))
    for bmark in eva_keys:
        # read selected frame

        # frame by frame visualization
        eva_annos = eva_anno_dict[args.dataset_name]

        anno_names = [f'{str(name)}.npz' for name in selected_frame_df.iloc[:, 0]]
        ftp_target_anno(eva_output, eva_annos[bmark], anno_names,
                    local_dir=os.path.join(args.cache_path, 'evaluation'))
        

        # every frame
    for idx, row in tqdm(selected_frame_df.iterrows(), desc=f'Processing {bmark}',
                            total=100, position=0, leave=False):
        anno_param_dict = {}
        for bmark in eva_keys:
            image_idx = row[0]
            image_path = row[1]
            anno_name = f'{str(image_idx)}.npz'

            # every benchmark
            target_local_file = os.path.join(args.cache_path,
                                'evaluation', eva_annos[bmark], anno_name)
            anno_param = dict(np.load(target_local_file, allow_pickle=True))
            anno_param_dict[bmark] = anno_param    

            # prepare args
            args.server_local_path = '/mnt/d/annotations_1988'
            args.image_cache_path = args.cache_path
            args.flat_hand_mean = False

        image_idxs.append(image_idx)
        image_paths.append(image_path)
        anno_param_dicts.append(anno_param_dict)

    # pdb.set_trace()

    print('Retriving annos from server...done')

    # pdb.set_trace()
    visualize_gt_eva(args, image_idxs, image_paths, anno_param_dicts)


if __name__ == '__main__':

    # main args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ehf')
    parser.add_argument('--out_path', type=str, 
                        default='/mnt/c/users/12595/desktop/humandata_vis/zzz-paper')

    # optional args
    parser.add_argument('--cache_path', type=str, default='/mnt/d/image_cache_1988',
                        required=False)
    args = parser.parse_args()

    visualize_eva(args)

    