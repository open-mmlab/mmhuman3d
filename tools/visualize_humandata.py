import numpy as np
import glob
import random
import cv2
import os
import argparse
import torch
import pyrender
import trimesh
import PIL.Image as pil_img

from mmhuman3d.models.body_models.builder import build_body_model

import pdb


def render_pose(img, body_model_param, body_model, camera):

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

    img = (output_img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img




def visualize_humandata(args):

    avaliable_datasets = ['shapy', 'gta_human2', 'renbody', 'egobody', 'ubody', 'ssp3d', 
                          'FIT3D', 'CHI3D', 'HumanSC3D', 'behave']

    dataset_path_dict = {
        'shapy': '/mnt/e/shapy',
        'gta_human2': '/mnt/e/gta_human2_multiple',
        'renbody': '/mnt/d/renbody',
        'egobody': '/mnt/d/egobody',
        'ubody': '/mnt/e/ubody',
        'ssp3d': '/mnt/e/ssp-3d',
        'FIT3D': '/mnt/d/sminchisescu-research-datasets',
        'CHI3D': '/mnt/d/sminchisescu-research-datasets',
        'HumanSC3D': '/mnt/d/sminchisescu-research-datasets',
        'behave': '/mnt/e/behave'}
    
    anno_path = {
        'shapy': 'output',
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
    param_ps = glob.glob(os.path.join(dataset_path_dict[dataset_name], 
                                      anno_path[dataset_name], f'{dataset_name}*.npz'))

    param = dict(np.load(param_ps[0], allow_pickle=True))

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
                    flat_hand_mean=True,
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
    
    idx_list = [100, 500, 1000, 500, 10000, 15000]

    # for idx in idx_list:
    for i in range(10):
        idx = random.randint(0, len(param['image_path']) - 1)

        image_p = param['image_path'][idx]

        # read image
        image_path = os.path.join(dataset_path_dict[dataset_name], image_p)
        image = cv2.imread(image_path)
        # convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

        # prepare for mesh projection
        camera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length[0], fy=focal_length[1],
            cx=camera_center[0], cy=camera_center[1])


        # read gender
        if has_gender:
            gender = param['meta'].item()['gender'][idx]
        else:
            gender = 'neutral'

        # read smpl smplx params and build body model
        if has_smpl:
            body_model_param_tensor = {key: torch.tensor(body_model_param_smpl[key][idx:idx+1], device=device)
                            for key in body_model_param_smpl.keys()}
            rendered_image = render_pose(img=image, 
                                        body_model_param=body_model_param_tensor, 
                                        body_model=smpl_model[gender],
                                        camera=camera)

        if has_smplx:
            body_model_param_tensor = {key: torch.tensor(body_model_param_smplx[key][idx:idx+1], device=device)
                            for key in body_model_param_smplx.keys()}
            rendered_image = render_pose(img=image, 
                                        body_model_param=body_model_param_tensor, 
                                        body_model=smplx_model[gender],
                                        camera=camera)

        os.makedirs(os.path.join(args.out_path, args.dataset_name), exist_ok=True)
        out_image_path = os.path.join(args.out_path, args.dataset_name,
                                    f'{os.path.basename(param_ps[0])[:-4]}_{idx}.png')
        print(f'Saving image to {out_image_path}')
        cv2.imwrite(out_image_path, rendered_image)

        # pdb.set_trace()



        continue



        vertices = output['vertices'].detach().cpu().numpy().squeeze()
        joints = output['joints'].detach().cpu().numpy().squeeze()  # in openpose topology
        body = trimesh.Trimesh(vertices, process=False)



        body_mesh = pyrender.Mesh.from_trimesh(body, material=material)

        pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        # image.shape = [9999, 9999]
        
        # build camera_pose and light
        camera_pose = np.eye(4)
        camera_pose = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1) * camera_pose
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)

        # build scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                       ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=pyrender2opencv)
        scene.add(light, pose=pyrender2opencv)
        scene.add(body_mesh, 'mesh')

        # render scene
        os.environ["PYOPENGL_PLATFORM"] = "osmesa" # include this line if use in vscode
        r = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                        viewport_height=image.shape[0],
                                        point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        alpha = 1.0  # set transparency in [0.0, 1.0]
        color[:, :, -1] = color[:, :, -1] * alpha
        pdb.set_trace()
        color = pil_img.fromarray((color * 255).astype(np.uint8))

        # concat on image
        output_img = pil_img.fromarray((image).astype(np.uint8))
        pdb.set_trace()
        output_img.paste(color, (0, 0, image.shape[1], image.shape[0]), color)
        # output_img.paste(color)

        # save image (downsize)
        output_img.convert('RGB')
        # output_img = output_img.resize((int(image.shape[1] / args.scale), int(image.shape[0] / args.scale)))
        output_img.save(os.path.join(args.out_path, 
                        f'{os.path.basename(param_ps[0])[:-4]}_{idx}.png'))

        pdb.set_trace()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=False, help='which dataset', default='renbody')
    # parser.add_argument('--dataset_path', type=str, required=True, help='path to the dataset')
    parser.add_argument('--out_path', type=str, required=False, help='path to the output folder',
                        default='/mnt/c/users/12595/desktop/humandata_vis')

    args = parser.parse_args()
    visualize_humandata(args)
    