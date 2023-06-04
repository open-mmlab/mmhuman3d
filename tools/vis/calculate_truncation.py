import numpy as np
import glob
import random
import cv2
import os
import argparse
import torch
import pyrender
import trimesh
import json
from tqdm import tqdm

from tools.convert_datasets import DATASET_CONFIGS
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx, get_keypoint_idxs_by_part
from mmhuman3d.models.body_models.builder import build_body_model

from tools.utils.request_files_server import request_files



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
    elif dataset_name in eft_datasets:
        pred_cam = param['meta'].item()['pred_cam'][idx]
        cam_scale, cam_trans = pred_cam[0], pred_cam[1:]
        # pdb.set_trace()
        bbox_xywh = param['bbox_xywh'][idx][:4]
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
                 R=None, T=None, dataset_name=None):

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    output = body_model(**body_model_param, return_verts=True)
    
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
    faces = body_model.faces

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
    output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)

    # output_img = color

    img = (output_img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8)

    return img


def render_truncation(img, body_model_param, body_model, cam_s, cam_b, cam_scale,
                      dataset_name=None, R=None, T=None):

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    output = body_model(**body_model_param, return_verts=True)
    
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
    faces = body_model.faces

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

    keypoints_3d = output['joints'].detach().cpu().numpy()

    # render material
    base_color = (1.0, 193/255, 193/255, 1.0)
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=base_color)

    # get body mesh
    body_trimesh = trimesh.Trimesh(vertices, faces, process=False)

    body_mesh_s = pyrender.Mesh.from_trimesh(body_trimesh, material=material)
    body_mesh_b = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    # project body centroid to image
    body_centroid = body_mesh_s.centroid
    # cam_s_intrinsics = cam_s.get_projection_matrix(img.shape[1], img.shape[0])
    cam_s_intrinsics = np.array([[cam_s.fx, 0, cam_s.cx],
                                 [0, cam_s.fy, cam_s.cy],
                                 [0, 0, 1]])
    body_centroid_s = cam_s_intrinsics @ body_centroid
    body_centroid_s = (body_centroid_s / body_centroid_s[-1])[:2]

    cam_b_intrinsics = np.array([[cam_b.fx, 0, cam_b.cx],
                                [0, cam_b.fy, cam_b.cy],
                                [0, 0, 1]])
    body_centroid_b = cam_b_intrinsics @ body_centroid
    body_centroid_b = (body_centroid_b / body_centroid_b[-1])[:2]

    # prepare camera and light
    light_s = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    light_b = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_pose = pyrender2opencv @ np.eye(4)
    
    # build scene_s
    scene_s = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))
    scene_s.add(cam_s, pose=cam_pose)
    scene_s.add(light_s, pose=cam_pose)
    scene_s.add(body_mesh_s, 'mesh')

    # render scene_s and get mask
    rs = pyrender.OffscreenRenderer(viewport_width=img.shape[1] * cam_scale,
                                    viewport_height=img.shape[0] * cam_scale,
                                    point_size=1.0)
    color_s, _ = rs.render(scene_s, flags=pyrender.RenderFlags.RGBA)
    color_s = color_s.astype(np.float32) / 255.0
    valid_mask_s = (color_s[:, :, -1] > 0)[:, :, np.newaxis]

    img_s = (color_s * 255).astype(np.uint8)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    # build scene_b
    scene_b = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))
    scene_b.add(cam_b, pose=cam_pose)
    scene_b.add(light_b, pose=cam_pose)
    scene_b.add(body_mesh_b, 'mesh')

    # render scene_b and get mask
    rb = pyrender.OffscreenRenderer(viewport_width=img.shape[1] * cam_scale,
                                    viewport_height=img.shape[0] * cam_scale,
                                    point_size=1.0)
    color_b, _ = rb.render(scene_b, flags=pyrender.RenderFlags.RGBA)
    color_b = color_b.astype(np.float32) / 255.0
    valid_mask_b = (color_b[:, :, -1] > 0)[:, :, np.newaxis]

    img_b = (color_b * 255).astype(np.uint8)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

    # # plot point on image
    # img_s = cv2.circle(img_s, (int(body_centroid_s[0]), int(body_centroid_s[1])), 10, (0, 0, 255), -1)
    # img_b = cv2.circle(img_b, (int(body_centroid_b[0]), int(body_centroid_b[1])), 10, (0, 0, 255), -1)

    # resize img_s to larger resolution
    img_s = cv2.resize(img_s, None, fx=cam_scale**2, fy=cam_scale**2, interpolation=cv2.INTER_NEAREST)
    valid_mask_s = (img_s[:, :, -1] > 0)[:, :, np.newaxis]
    # adjust body centroid accordingly
    body_centroid_s = (body_centroid_s * cam_scale**2).astype(np.int32)

    # align left top corner of img_b to img_s
    body_centroid_b = body_centroid_b.astype(np.int32)
    img_b_lefttop = body_centroid_s - body_centroid_b

    # pad image_b for overlay, note that ima_b and img_b_lefttop are in different (W, H) order
    img_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR)
    img_s = np.pad(img_b, ((img_b_lefttop[1], img_s.shape[0] - img_b.shape[0] - img_b_lefttop[1]), 
                           (img_b_lefttop[0], img_s.shape[1] - img_b.shape[1] - img_b_lefttop[0]), (0, 0)), 
                           mode='constant')+ img_s

    # calculate truncation
    in_screen_mask_s = valid_mask_s[img_b_lefttop[1]:img_b_lefttop[1] + img_b.shape[0] +1,
                                    img_b_lefttop[0]:img_b_lefttop[0] + img_b.shape[1] +1,
                                    :]
    # in_screen_mask_s = valid_mask_s[:img_b.shape[0],:img_b.shape[1],:]
    # in_screen_mask_s = np.pad(in_screen_mask_s, ((0, padding_width), (0, padding_height), (0, 0)), mode='constant')
    
    if np.sum(valid_mask_s) == 0:
        trunc = 1
        return img_s, trunc
    else:
        trunc = 1 - np.sum(in_screen_mask_s) / np.sum(valid_mask_s)

        # crop image_s to its bbox
        rows, cols, _ = np.where(valid_mask_s)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        cropped_image = img_s[min_row:max_row+1, min_col:max_col+1, :]
        return cropped_image, trunc

def visualize_humandata(args):

    avaliable_datasets = ['shapy', 'gta_human2', 'egobody', 'ubody', 'ssp3d', 
                          'FIT3D', 'CHI3D', 'HumanSC3D', 'behave']
    # some datasets are on 1988
    server_datasets = ['arctic', 'bedlam', 'crowdpose', 'lspet', 'ochuman',
                       'posetrack', 'ehf', 'instavariety', 'mpi_inf_3dhp',
                       'mtp', 'muco3dhp', 'prox', 'renbody', 'rich', 'spec',
                       'synbody','talkshow', 'up3d', 'renbody_highres', 'agora']

    
    humandata_datasets = [ # name, glob pattern, exclude pattern
        ('arctic', 'p1_train.npz', ''),
        ('bedlam', 'bedlam_train.npz', ''),
        ('behave', 'behave_train_230516_231_downsampled.npz', ''),
        ('chi3d', 'CHI3D_train_230511_1492_*.npz', ''),
        ('crowdpose', 'crowdpose_neural_annot_train_new.npz', ''),
        ('lspet', 'eft_lspet.npz', ''),
        ('ochuman', 'eft_ochuman.npz', ''),
        ('posetrack', 'eft_posetrack.npz', ''),
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
        ('renbody_highres', 'renbody_train_highrescam_230517_399_*_fix_betas.npz', ''),
        ('rich', 'rich_train_fix_betas.npz', ''),
        ('spec', 'spec_train_smpl.npz', ''),
        ('ssp3d', 'ssp3d_230525_311.npz', ''),
        ('synbody_magic1', 'synbody_amass_230328_02172.npz', ''),
        ('synbody', 'synbody_train_230521_04000_fix_betas.npz', ''),
        ('talkshow', 'talkshow_smplx_*.npz', 'path'),
        ('up3d', 'up3d_trainval.npz', ''),
    ]

    dataset_path_dict = {
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
        'ehf': '',
        'HumanSC3D': '/mnt/d/sminchisescu-research-datasets',
        'instavariety': '/lustrenew/share_data/zoetrope/data/datasets/neural_annot_data/insta_variety/',
        'mpi_inf_3dhp': '/lustrenew/share_data/zoetrope/data/datasets/neural_annot_data/mpi_inf_3dhp',
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
        'ubody': 'output',
        'ssp3d': 'output',
        'FIT3D': 'output',
        'CHI3D': 'output',
        'HumanSC3D': 'output',
        'behave': 'output'}
    
    camera_params_dict = {
        'gta_human2': {'principal_point': [960, 540], 'focal_length': [1158.0337, 1158.0337]},
        'ssp3d': {'focal_length': [5000, 5000], 'principal_point': [256, 256]},
        'synbody': {'focal_length': [640, 640], 'principal_point': [640, 360]},
    }

    # load humandata
    dataset_name = args.dataset_name
    flat_hand_mean = False

    if dataset_name not in server_datasets:
        param_ps = glob.glob(os.path.join(dataset_path_dict[dataset_name], 
                                            anno_path[dataset_name], f'*.npz'))
        param_ps = [p for p in param_ps if 'test' not in p]
        param_ps = [p for p in param_ps if 'val' not in p]
    else:
        data_info = [info for info in humandata_datasets if info[0] == dataset_name][0]
        _, filename, exclude = data_info

        param_ps = glob.glob(os.path.join(args.server_local_path, filename))
        if exclude != '':
            param_ps = [p for p in param_ps if exclude not in p]
    # truncation dict
    trunc_dict = {}

    for npz_id, param_p in enumerate(tqdm(param_ps, desc=f'Processing npzs',
                        position=0, leave=False)):
        
        trunc_log = {'image_path': [], 'truncation': []}

        param = dict(np.load(param_p, allow_pickle=True))

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
                        flat_hand_mean=False,
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
                        flat_hand_mean=flat_hand_mean,
                        use_pca=False,
                        batch_size=1
                    )).to(device)
        
        # for idx in idx_list:
        sample_size =  args.sample_size
        if sample_size > len(param['image_path']):
            idxs = range(len(param['image_path']))
        else:
            idxs = random.sample(range(len(param['image_path'])), sample_size)

        # idxs = [340, 139181]

        # prepare for server request if datasets on server
        if dataset_name in server_datasets:
            print('getting images from server...')
            files = param['image_path'][idxs].tolist()
            local_image_folder = os.path.join(args.image_cache_path, dataset_name)
            os.makedirs(local_image_folder, exist_ok=True)
            request_files(files, 
                        server_path=dataset_path_dict[dataset_name], 
                        local_path=local_image_folder, 
                        server_name='1988')
            print('done')
        else:
            local_image_folder = dataset_path_dict[dataset_name]
        

        for idx in tqdm(idxs, desc=f'Processing npzs {npz_id}/{len(param_ps)}, sample size: {sample_size}',
                        position=1, leave=False):
            image_p = param['image_path'][idx]
            # read image
            # pdb.set_trace()
            image_path = os.path.join(local_image_folder, image_p)
            image = cv2.imread(image_path)
            # convert to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

            # ---------------------- render pose ----------------------
            # prepare for mesh projection
            camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0], fy=focal_length[1],
                cx=camera_center[0], cy=camera_center[1])
            
            # ---------------------- render truncation ---------------------- 
            # prepare camera for calculating truncation and pp occlusion
            cam_scale = 2
            camera_small = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0] / cam_scale, fy=focal_length[1] / cam_scale,
                cx=camera_center[0] * cam_scale, cy=camera_center[1] * cam_scale)
            camera_big = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0] * cam_scale, fy=focal_length[1] * cam_scale,
                cx=camera_center[0] * cam_scale, cy=camera_center[1] * cam_scale)
            
            if dataset_name in ['ochuman']:
                camera = pyrender.camera.IntrinsicsCamera(
                    fx=focal_length[0], fy=focal_length[1],
                    cx=camera_center[0], cy=camera_center[1],
                    zfar=600, znear=30)                    
                camera_small = pyrender.camera.IntrinsicsCamera(
                    fx=focal_length[0] / cam_scale, fy=focal_length[1] / cam_scale,
                    cx=camera_center[0] * cam_scale, cy=camera_center[1] * cam_scale,
                    zfar=600, znear=30)
                camera_big = pyrender.camera.IntrinsicsCamera(
                    fx=focal_length[0] * cam_scale, fy=focal_length[1] * cam_scale,
                    cx=camera_center[0] * cam_scale, cy=camera_center[1] * cam_scale,
                    zfar=600, znear=30)

            # read smpl smplx params and build body model
            if has_smpl:
                intersect_key = list(set(body_model_param_smpl.keys()) & set(smpl_shape.keys()))
                body_model_param_tensor = {key: torch.tensor(
                        np.array(body_model_param_smpl[key][idx:idx+1]).reshape(smpl_shape[key]),
                                device=device, dtype=torch.float32)
                                for key in intersect_key
                                if len(body_model_param_smpl[key][idx:idx+1]) > 0}
                rendered_image = render_pose(img=image, 
                                            body_model_param=body_model_param_tensor, 
                                            body_model=smpl_model[gender],
                                            camera=camera,
                                            dataset_name=dataset_name,
                                            R=R, T=T)
                trunc_image, trunc = render_truncation(img=image,
                                            body_model_param=body_model_param_tensor, 
                                            body_model=smpl_model[gender],
                                            cam_s=camera_small,
                                            cam_b=camera_big,
                                            cam_scale=cam_scale,
                                            dataset_name=dataset_name,
                                            R=R, T=T)

            if has_smplx:
                intersect_key = list(set(body_model_param_smplx.keys()) & set(smplx_shape.keys()))
                body_model_param_tensor = {key: torch.tensor(
                        np.array(body_model_param_smplx[key][idx:idx+1]).reshape(smplx_shape[key]),
                                device=device, dtype=torch.float32)
                                for key in intersect_key
                                if len(body_model_param_smplx[key][idx:idx+1]) > 0}
                rendered_image = render_pose(img=image, 
                                            body_model_param=body_model_param_tensor, 
                                            body_model=smplx_model[gender],
                                            camera=camera,
                                            dataset_name=dataset_name,
                                            R=R, T=T)
                trunc_image, trunc = render_truncation(img=image,
                                            body_model_param=body_model_param_tensor, 
                                            body_model=smplx_model[gender],
                                            cam_s=camera_small,
                                            cam_b=camera_big,
                                            cam_scale=cam_scale,
                                            dataset_name=dataset_name,
                                            R=R, T=T)
                
            trunc_log['image_path'].append(image_p)
            trunc_log['truncation'].append(trunc)
                
            # ---------------------- render results ----------------------
            # print(f'Truncation: {trunc}')
            os.makedirs(os.path.join(args.out_path, dataset_name), exist_ok=True)

            # for writting on image
            font_size = image.shape[1] / 1000
            line_sickness = int(image.shape[1] / 1000) + 1
            front_location_y = int(image.shape[1] / 10)
            front_location_x = int(image.shape[0] / 10)

            if args.save_trunc:
                cv2.putText(rendered_image, f'Truncation: {trunc}', 
                        (front_location_x, front_location_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_size, (0, 0, 255), line_sickness, cv2.LINE_AA)
                
                out_image_path_trunc = os.path.join(args.out_path, dataset_name,
                                    f'{os.path.basename(param_p)[:-4]}_{idx}_trunc.png')
                cv2.imwrite(out_image_path_trunc, trunc_image)

                out_image_path = os.path.join(args.out_path, dataset_name,
                                            f'{os.path.basename(param_p)[:-4]}_{idx}.png')
                cv2.imwrite(out_image_path, rendered_image)

            # pdb.set_trace()

        trunc_dict[os.path.basename(param_p)] = trunc_log
    
    # save truncation dict
    with open(os.path.join(args.out_path, f'{args.dataset_name}_truncation.json'), 'w') as f:
        json.dump(trunc_dict, f)


if __name__ == '__main__':

    # python tools/vis/calculate_truncation.py --dataset_name egobody

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=False, 
                        help='which dataset', 
                        default='ubody')
    # parser.add_argument('--dataset_path', type=str, required=True, help='path to the dataset')
    parser.add_argument('--out_path', type=str, required=False, 
                        help='path to the output folder',
                        default='/mnt/c/users/12595/desktop/humandata_vis/zzz-truncation')
    parser.add_argument('--server_local_path', type=str, required=False, 
                        help='local path to where you save the annotations of server datasets',
                        default='/mnt/d/annotations_1988')
    parser.add_argument('--image_cache_path', type=str, required=False,
                        help='local path to image cache folder for server datasets',
                        default='/mnt/d/image_cache_1988')
    

    # optional args
    parser.add_argument('--save_trunc', type=bool, required=False, 
                        help='save truncation images', 
                        default=True)
    parser.add_argument('--sample_size', type=int, required=False,
                        help='number of samples to visualize',
                        default=1000)

    args = parser.parse_args()
    visualize_humandata(args)
    