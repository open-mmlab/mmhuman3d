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

import pdb


def get_cam_params(camera_params_dict, dataset_name, param, idx):

    # read cam params
    if dataset_name in camera_params_dict.keys():
        cx, cy = camera_params_dict[dataset_name]['principal_point']
        fx, fy = camera_params_dict[dataset_name]['focal_length']
        camera_center = (cx, cy)
        focal_length = (fx, fy)
    else:
        try:
            focal_length = param['meta'].item()['focal_length'][idx]
            camera_center = param['meta'].item()['principal_point'][idx]
        except KeyError:
            focal_length = param['misc'].item()['focal_length']
            camera_center = param['meta'].item()['principal_point']
    
    if isinstance(focal_length, float):
        focal_length = [focal_length, focal_length]
    if isinstance(camera_center, float):
        camera_center = [camera_center, camera_center]

    return focal_length, camera_center


def render_pose(img, body_model_param, body_model, camera, return_mask=False):

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    output = body_model(**body_model_param, return_verts=True)
    
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
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
    output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)

    # output_img = color

    img = (output_img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8)

    return img


def render_truncation(img, body_model_param, body_model, cam_s, cam_b, cam_scale):

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    output = body_model(**body_model_param, return_verts=True)
    
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
    faces = body_model.faces

    keypoints_3d = output['joints'].detach().cpu().numpy()
    pelvis_pyrender = keypoints_3d[0, 0]

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
    trunc = 1 - np.sum(in_screen_mask_s) / np.sum(valid_mask_s)

    # crop image_s to its bbox
    rows, cols, _ = np.where(valid_mask_s)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    cropped_image = img_s[min_row:max_row+1, min_col:max_col+1, :]

    return cropped_image, trunc


def visualize_humandata(args):

    avaliable_datasets = ['shapy', 'gta_human2', 'renbody', 'egobody', 'ubody', 'ssp3d', 
                          'FIT3D', 'CHI3D', 'HumanSC3D', 'behave']

    dataset_path_dict = {
        'shapy': '/mnt/e/shapy',
        'gta_human2': '/mnt/e/gta_human2',
        'renbody': '/mnt/d/renbody',
        'egobody': '/mnt/d/egobody',
        'ubody': '/mnt/d/ubody',
        'ssp3d': '/mnt/e/ssp-3d',
        'FIT3D': '/mnt/d/sminchisescu-research-datasets',
        'CHI3D': '/mnt/d/sminchisescu-research-datasets',
        'HumanSC3D': '/mnt/d/sminchisescu-research-datasets',
        'behave': '/mnt/e/behave'}
    
    anno_path = {
        'shapy': 'output',
        'gta_human2': 'output',
        'renbody': 'output',
        'egobody': 'output',
        'ubody': 'output',
        'ssp3d': 'output',
        'FIT3D': 'output',
        'CHI3D': 'output',
        'HumanSC3D': 'output',
        'behave': 'output'}
    
    camera_params_dict = {
        'gta_human2': {'principal_point': [960, 540], 'focal_length': [1158.0337, 1158.0337]},

    }

    # load humandata
    dataset_name = args.dataset_name

    if 'modes' in DATASET_CONFIGS[dataset_name].keys(): 
        dataset_modes = DATASET_CONFIGS[dataset_name]['modes']

        # only use train modes
        dataset_modes = [mode for mode in dataset_modes if 'test' not in mode]
        dataset_modes = [mode for mode in dataset_modes if 'val' not in mode]

        for mode in dataset_modes:
            param_ps = glob.glob(os.path.join(dataset_path_dict[dataset_name], 
                                        anno_path[dataset_name], f'{dataset_name}*{mode}*.npz'))        

    else:
        param_ps = glob.glob(os.path.join(dataset_path_dict[dataset_name], 
                                        anno_path[dataset_name], f'{dataset_name}*.npz'))
        
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
                        flat_hand_mean=False,
                        use_pca=False,
                        batch_size=1
                    )).to(device)
        
        # idx_list = [340, 139181]

        # for idx in idx_list:
        if args.sample_size > len(param['image_path']):
            idxs = range(len(param['image_path']))
        else:
            idxs = random.sample(range(len(param['image_path'])), args.sample_size)

        for i in tqdm(idxs, desc=f'Processing {args.sample_size} sample, '
                      f'in {os.path.basename(param_p)}',
                        position=1, leave=False):
            idx = random.randint(0, len(param['image_path']) - 1)

            image_p = param['image_path'][idx]

            # read image
            image_path = os.path.join(dataset_path_dict[dataset_name], image_p)
            image = cv2.imread(image_path)
            # convert to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # read cam params
            focal_length, camera_center = get_cam_params(camera_params_dict, args.dataset_name, param, idx)
    
            # read gender
            if has_gender:
                gender = param['meta'].item()['gender'][idx]
            else:
                gender = 'neutral'

            # ---------------------- render pose ----------------------
            # prepare for mesh projection
            camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0], fy=focal_length[1],
                cx=camera_center[0], cy=camera_center[1])
            
            # read smpl smplx params and build body model
            if has_smpl:
                body_model_param_tensor = {key: torch.tensor(body_model_param_smpl[key][idx:idx+1],
                                                            device=device, dtype=torch.float32)
                                for key in body_model_param_smpl.keys()
                                if body_model_param_smpl[key][idx:idx+1].shape[0] > 0}
                rendered_image = render_pose(img=image, 
                                            body_model_param=body_model_param_tensor, 
                                            body_model=smpl_model[gender],
                                            camera=camera)

            if has_smplx:
                body_model_param_tensor = {key: torch.tensor(body_model_param_smplx[key][idx:idx+1],
                                                            device=device, dtype=torch.float32)
                                for key in body_model_param_smplx.keys()
                                if body_model_param_smplx[key][idx:idx+1].shape[0] > 0}
                rendered_image = render_pose(img=image, 
                                            body_model_param=body_model_param_tensor, 
                                            body_model=smplx_model[gender],
                                            camera=camera)

            # ---------------------- render truncation ---------------------- 
            # prepare camera for calculating truncation and pp occlusion
            cam_scale = 2
            camera_small = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0] / cam_scale, fy=focal_length[1] / cam_scale,
                cx=camera_center[0] * cam_scale, cy=camera_center[1] * cam_scale)
            camera_big = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0] * cam_scale, fy=focal_length[1] * cam_scale,
                cx=camera_center[0] * cam_scale, cy=camera_center[1] * cam_scale)

            # read smpl smplx params and build body model
            if has_smpl:
                body_model_param_tensor = {key: torch.tensor(body_model_param_smpl[key][idx:idx+1],
                                                            device=device, dtype=torch.float32)
                                for key in body_model_param_smpl.keys()
                                if body_model_param_smpl[key][idx:idx+1].shape[0] > 0}
                trunc_image, trunc = render_truncation(img=image,
                                            body_model_param=body_model_param_tensor, 
                                            body_model=smpl_model[gender],
                                            cam_s=camera_small,
                                            cam_b=camera_big,
                                            cam_scale=cam_scale)

            if has_smplx:
                body_model_param_tensor = {key: torch.tensor(body_model_param_smplx[key][idx:idx+1],
                                                            device=device, dtype=torch.float32)
                                for key in body_model_param_smplx.keys()
                                if body_model_param_smplx[key][idx:idx+1].shape[0] > 0}
                trunc_image, trunc = render_truncation(img=image,
                                            body_model_param=body_model_param_tensor, 
                                            body_model=smplx_model[gender],
                                            cam_s=camera_small,
                                            cam_b=camera_big,
                                            cam_scale=cam_scale)

            trunc_log['image_path'].append(image_p)
            trunc_log['truncation'].append(trunc)
                
            # ---------------------- render results ----------------------
            # print(f'Truncation: {trunc}')
            os.makedirs(os.path.join(args.out_path, args.dataset_name), exist_ok=True)

            # for writting on image
            font_size = image.shape[1] / 1000
            line_sickness = int(image.shape[1] / 1000) + 1
            front_location_y = int(image.shape[1] / 10)
            front_location_x = int(image.shape[0] / 10)

            if args.save_trunc:
                cv2.putText(rendered_image, f'Truncation: {trunc}', 
                        (front_location_x, front_location_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_size, (0, 0, 255), line_sickness, cv2.LINE_AA)
                
                out_image_path_trunc = os.path.join(args.out_path, args.dataset_name,
                                    f'{os.path.basename(param_p)[:-4]}_{idx}_trunc.png')
                cv2.imwrite(out_image_path_trunc, trunc_image)

                out_image_path = os.path.join(args.out_path, args.dataset_name,
                                            f'{os.path.basename(param_p)[:-4]}_{idx}.png')
                cv2.imwrite(out_image_path, rendered_image)

            # pdb.set_trace()

        trunc_dict[os.path.basename(param_ps[0])] = trunc_log
    
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
    

    # optional args
    parser.add_argument('--save_trunc', type=bool, required=False, 
                        help='save truncation images', 
                        default=True)
    parser.add_argument('--sample_size', type=int, required=False,
                        help='number of samples to visualize',
                        default=1000)

    args = parser.parse_args()
    visualize_humandata(args)
    