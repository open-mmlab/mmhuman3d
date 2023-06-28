import argparse
import glob
import os
import pdb
import random

import cv2
import numpy as np
import PIL.Image as pil_img
import pyrender
import smplx
import torch
import trimesh
from tools.convert_datasets import DATASET_CONFIGS
from tools.utils.request_files_server import request_files, request_files_name
from tools.vis.visualize_humandata_qp import render_pose as render_pose_qp
from tools.vis.visualize_humandata_seq import (
    dataset_path_dict,
    humandata_datasets,
)
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.body_models.utils import transform_to_camera_frame
from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa

smpl_shape = {
    'betas': (-1, 10),
    'transl': (-1, 3),
    'global_orient': (-1, 3),
    'body_pose': (-1, 69)
}


def render_pose(
    img,
    body_model_param,
    body_model,
    camera,
    return_mask=False,
):

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                                [0, 0, 0, 1]])

    output = body_model(**body_model_param, return_verts=True)

    vertices = output['vertices'].detach().cpu().numpy().squeeze()

    # # overlay on img
    # verts = output['vertices']
    # verts2d = camera_opencv.transform_points_screen(verts)[..., :2].detach().cpu().numpy()

    # for i in range(verts2d.shape[1]):
    #     cv2.circle(img, (int(verts2d[0, i, 0]), int(verts2d[0, i, 1])), 1, (0, 255, 0), -1)

    faces = body_model.faces

    # render material
    base_color = (1.0, 193 / 255, 193 / 255, 1.0)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=base_color)

    # get body mesh
    body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    # project body centroid to screen (for test)
    body_centroid = body_mesh.centroid
    cam_intrinsics = np.array([[camera.fx, 0, camera.cx],
                               [0, camera.fy, camera.cy], [0, 0, 1]])
    body_centroid = cam_intrinsics @ body_centroid
    body_centroid_2d = (body_centroid / body_centroid[-1])[:2]

    # prepare camera and light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_pose = pyrender2opencv @ np.eye(4)

    # build scene
    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=cam_pose)
    scene.add(body_mesh, 'mesh')

    # render scene
    os.environ[
        'PYOPENGL_PLATFORM'] = 'osmesa'  # include this line if use in vscode
    r = pyrender.OffscreenRenderer(
        viewport_width=img.shape[1],
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
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8)

    return img, body_centroid_2d


def solvepnp(args):

    dataset_name = args.dataset_name
    dataset_path = f'/mnt/e/{dataset_name}'
    device = torch.device('cuda:0')

    # load dataset
    dp = glob.glob(os.path.join(dataset_path, '*.npz'))[0]
    params = dict(np.load(dp, allow_pickle=True))

    dataset_size = len(params['image_path'])

    # init focal length and camera center for refine meta
    meta_data = params['meta'].item()
    focal_lengths, camera_centers = [], []

    for i in tqdm(range(dataset_size)):
        # test on a single image
        # img_idx = random.randint(0, len(params['image_path'])-1)
        img_idx = i
        image_path = params['image_path'][img_idx]

        # load image
        image = cv2.imread(os.path.join(dataset_path, image_path))

        # camera
        camera_param_dict = params['meta'].item()
        height = camera_param_dict['height'][img_idx]
        width = camera_param_dict['width'][img_idx]
        pred_cam = camera_param_dict['pred_cam'][img_idx]

        # kps2d
        kps2d_orig = params['keypoints2d_ori'][img_idx]
        kps2d = params['keypoints2d_smpl'][img_idx].reshape(1, -1, 3)
        kps2d, _ = convert_kps(kps2d, 'smpl_54', 'smpl_45')
        kps2d = kps2d.reshape(-1, 3)

        # overlay keypoints original 2d
        image_kps = image.copy()
        for i in range(kps2d_orig.shape[0]):
            cv2.circle(image_kps, tuple(kps2d_orig[i][:2].astype(int)), 3,
                       (0, 0, 255), -1)
            # write kps index
            cv2.putText(image_kps, str(i),
                        tuple(kps2d_orig[i][:2].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        cv2.LINE_AA)

        # overlay keypoints 2d
        for i in range(kps2d.shape[0]):
            cv2.circle(image_kps, tuple(kps2d[i][:2].astype(int)), 3,
                       (255, 0, 0), -1)
            # write kps index
            cv2.putText(image_kps, str(i), tuple(kps2d[i][:2].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                        cv2.LINE_AA)

        image_outpath = os.path.join(dataset_path, 'output',
                                     os.path.basename(image_path))
        # cv2.imwrite(image_outpath, image_kps)

        # read smpl parameters
        smpl_param = params['smpl'].item()
        intersect_key = list(set(smpl_param.keys()) & set(smpl_shape.keys()))
        body_model_param = {
            key: np.array(smpl_param[key][img_idx:img_idx + 1]).reshape(
                smpl_shape[key])
            for key in intersect_key
            if len(smpl_param[key][img_idx:img_idx + 1]) > 0
        }
        body_model_param_tensor = {
            key: torch.tensor(
                body_model_param[key], device=device, dtype=torch.float32)
            for key in intersect_key
        }

        # create smpl model
        smpl_model = build_body_model(
            dict(
                type='SMPL',
                keypoint_src='smpl_45',
                keypoint_dst='smpl_45',
                model_path='data/body_models/smpl',
                gender='neutral',
                num_betas=10,
                use_face_contour=True,
                use_pca=False,
                batch_size=1)).to(device)
        output = smpl_model(return_verts=True, **body_model_param_tensor)
        kps3d_smpl = output['joints'].detach().cpu().numpy().squeeze()

        # match kps2d and kps3d
        ochuman_mapping = [
            [0, 4],
            [1, 5],
            [4, 16],
            [5, 17],
            [8, 20],
            [9, 21],
            [11, 23],
            [12, 24],
        ]
        objPoints = []
        imgPoints = []
        for i in range(len(ochuman_mapping)):
            mapping = ochuman_mapping[i]
            if 0 in kps2d_orig[mapping[0]]:
                continue
            imgPoints.append(kps2d_orig[mapping[0]][:2])
            objPoints.append(kps3d_smpl[mapping[1]][:3])

        # # match kps2d_smpl and kps3d
        # imgPoints = kps2d[..., :2].reshape(-1, 2)
        # objPoints = kps3d_smpl.reshape(-1, 3)

        # assume a prespective camera
        focal_length = [5000, 5000]
        camera_center = [width / 2, height / 2]

        # create camera matrix
        cameraMatrix = np.zeros((3, 3))
        cameraMatrix[0, 0] = focal_length[0]
        cameraMatrix[1, 1] = focal_length[1]
        cameraMatrix[0, 2] = camera_center[0]
        cameraMatrix[1, 2] = camera_center[1]

        # opencv solvepnp (assume no distortion)
        objPoints = np.array(objPoints).reshape(-1, 3)
        imgPoints = np.array(imgPoints).reshape(-1, 2)
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            objPoints, imgPoints, cameraMatrix, distCoeffs=None, flags=1)
        # print(f'rotation: {rvec} \n translation: {tvec}')
        # if tvec[2] < 0:
        #     tvec = -tvec

        # create camera extrinsic matrix
        extrinics = np.eye(4)
        cam_rotation = aa_to_rotmat(rvec.reshape(1, -1)).reshape(3, 3)
        # cam_rotation = np.linalg.inv(cam_rotation)
        extrinics[:3, :3] = cam_rotation
        tvec = tvec * 1
        extrinics[:3, 3] = tvec.reshape(3)

        # transfrom smpl to camera space
        pelvis_world = kps3d_smpl[get_keypoint_idx('pelvis', 'smpl')]
        global_orient_cam, transl_cam = transform_to_camera_frame(
            global_orient=body_model_param['global_orient'],
            transl=[0, 0, 0],
            pelvis=pelvis_world,
            extrinsic=extrinics)

        # write to body model param
        # body_model_param['global_orient'] = global_orient_cam.reshape(-1, 3)
        body_model_param['transl'] = transl_cam.reshape(-1, 3)

        body_model_param_tensor = {
            key: torch.tensor(
                body_model_param[key], device=device, dtype=torch.float32)
            for key in body_model_param.keys()
        }

        # prepare camera
        camera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length[0],
            fy=focal_length[1],
            cx=camera_center[0],
            cy=camera_center[1],
        )

        # render image
        rendered_image, body_centroid = render_pose(
            img=image,
            body_model_param=body_model_param_tensor,
            body_model=smpl_model,
            camera=camera)
        # print(body_centroid)

        # save rendered image
        image_outpath = os.path.join(
            dataset_path, 'output',
            os.path.basename(image_path).replace('.jpg', '_rendered_pnp.jpg'))
        cv2.imwrite(image_outpath, rendered_image)

        # append focal length and camera center
        focal_lengths.append(focal_length)
        camera_centers.append(camera_center)

    # load and refine humandata
    human_data = HumanData.fromfile(dp)

    meta_data['focal_length'] = focal_lengths
    meta_data['camera_center'] = camera_centers
    human_data['meta'] = meta_data

    # save npz
    refined_npzp = f'{dp[-4]}_refined_camera.npz'
    human_data.dump(refined_npzp)

    # pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=False,
        help='which dataset',
        default='ochuman')
    parser.add_argument(
        '--num_samples',
        type=int,
        required=False,
        help='number of samples to test',
        default=10)

    args = parser.parse_args()

    RT_datasets = ['ochuman']

    solvepnp(args)
