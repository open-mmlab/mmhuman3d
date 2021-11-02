import copy
import os
import os.path as osp
import shutil
import warnings
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from configs.render.smpl import RENDER_CONFIGS
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmhuman3d.core.cameras import (
    WeakPerspectiveCameras,
    compute_orbit_cameras,
)
from mmhuman3d.core.conventions.cameras import convert_cameras
from mmhuman3d.utils.ffmpeg_utils import (
    images_to_array,
    video_to_array,
    video_to_images,
)
from mmhuman3d.utils.path_utils import (
    check_input_path,
    check_path_suffix,
    prepare_output_path,
)
from mmhuman3d.utils.smpl_utils import get_body_model, get_mesh_info
from .renderer import RenderDataset, SMPLRenderer

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _prepare_mesh(poses, betas, transl, verts, start, end, body_model):
    # check the format of input poses.
    NUM_JOINTS = body_model.NUM_JOINTS
    NUM_BODY_JOINTS = body_model.NUM_BODY_JOINTS
    NUM_DIM = 3 * (NUM_JOINTS + 1)
    body_pose_keys = body_model.body_pose_keys
    joints = None
    if poses is not None:
        if isinstance(poses, dict):
            if not body_pose_keys.issubset(poses):
                raise KeyError(
                    f'{str(poses.keys())}, Please make sure that your '
                    f'input dict has all of {", ".join(body_pose_keys)}')
            num_frame = poses['body_pose'].shape[0]
            _, num_person, _ = poses['body_pose'].view(
                num_frame, -1, NUM_BODY_JOINTS * 3).shape

            full_pose = body_model.dict2tensor(poses)
            start = (min(start, num_frame - 1) + num_frame) % num_frame
            end = (min(end, num_frame - 1) + num_frame) % num_frame
            full_pose = full_pose[start:end + 1]

        elif isinstance(poses, torch.Tensor):
            if poses.shape[-1] != NUM_DIM:
                raise ValueError(
                    f'Please make sure your poses is {NUM_DIM} dims in'
                    f'the last axis. Your input shape: {poses.shape}')
            poses = poses.view(poses.shape[0], -1, (NUM_JOINTS + 1) * 3)
            num_frame, num_person, _ = poses.shape
            start = (min(start, num_frame - 1) + num_frame) % num_frame
            end = (min(end, num_frame - 1) + num_frame) % num_frame
            full_pose = poses[start:end + 1]
        else:
            raise ValueError('Wrong pose type, should be `dict` or `tensor`.')
        # slice the input poses, betas, and transl.
        betas = betas[start:end + 1] if betas is not None else None
        transl = transl[start:end + 1] if transl is not None else None
        pose_dict = body_model.tensor2dict(
            full_pose=full_pose, betas=betas, transl=transl)

        # get new num_frame
        num_frame = full_pose.shape[0]

        mesh_info = get_mesh_info(
            body_model=body_model,
            data_type='tensor',
            required_keys=['vertices', 'faces', 'joints'],
            **pose_dict)

        vertices, faces, joints = mesh_info['vertices'], mesh_info[
            'faces'], mesh_info['joints']

    elif verts is not None:
        if isinstance(verts, np.ndarray):
            verts = torch.Tensor(verts)

        pose_dict = body_model.tensor2dict(
            torch.zeros(1, (NUM_JOINTS + 1) * 3))
        mesh_info = get_mesh_info(
            body_model=body_model,
            data_type='tensor',
            required_keys=['vertices', 'faces'],
            **pose_dict)
        num_verts = mesh_info['vertices'].shape[-2]
        assert verts.shape[-2] == num_verts, 'Wrong input verts shape.'
        faces = mesh_info['faces']
        num_frame = verts.shape[0]
        start = (min(start, num_frame - 1) + num_frame) % num_frame
        end = (min(end, num_frame - 1) + num_frame) % num_frame
        verts = verts[start:end + 1]
        num_frame = verts.shape[0]
        vertices = verts.view(num_frame, -1, num_verts, 3)
        num_person = vertices.shape[1]
    else:
        raise ValueError('Poses and verts are all None.')
    return vertices, faces, joints, num_frame, num_person, start, end


def render_smpl(
    # smpl parameters
    poses: Optional[Union[torch.Tensor, np.ndarray, dict]] = None,
    betas: Optional[Union[torch.Tensor, np.ndarray]] = None,
    transl: Optional[Union[torch.Tensor, np.ndarray]] = None,
    verts: Optional[Union[torch.Tensor, np.ndarray]] = None,
    model_type: Literal['smpl', 'smplx'] = 'smpl',
    gender: Literal['male', 'female', 'neutral'] = 'neutral',
    body_model_dir: Optional[str] = None,
    body_model: Optional[nn.Module] = None,
    use_pca: bool = False,
    # camera paramters
    R: Optional[Union[torch.Tensor, np.ndarray]] = None,
    T: Optional[Union[torch.Tensor, np.ndarray]] = None,
    K: Optional[Union[torch.Tensor, np.ndarray]] = None,
    pred_cam: Optional[Union[torch.Tensor, np.ndarray]] = None,
    in_ndc: bool = True,
    convention: str = 'pytorch3d',
    projection: Literal['weakperspective', 'perspective', 'fovperspective',
                        'orthographics', 'fovorthographics'] = 'perspective',
    orbit_speed: Union[float, Tuple[float, float]] = 0.0,
    # render choice parameters
    render_choice: Literal['lq', 'mq', 'hq', 'silhouette', 'depth',
                           'pointcloud', 'part_silhouette'] = 'hq',
    palette: Union[List[str], str, np.ndarray] = 'white',
    resolution: Union[List[int], Tuple[int, int]] = (1024, 1024),
    start: int = 0,
    end: int = -1,
    alpha: float = 1.0,
    no_grad: bool = False,
    batch_size: int = 20,
    DDP: bool = False,
    # file io parameters
    return_tensor: bool = False,
    output_path: str = None,
    origin_frames: Optional[str] = None,
    frame_list: Optional[List[str]] = None,
    image_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
    img_format: str = 'frame_%06d.jpg',
    overwrite: bool = False,
    obj_path: Optional[str] = None,
    read_frames_batch: bool = False,
    # visualize keypoints
    plot_kps: bool = False,
    kp3d: Optional[Union[np.ndarray, torch.Tensor]] = None,
    smpl_joints: Optional[Union[np.ndarray, torch.Tensor]] = None,
    mask: Optional[Union[np.ndarray, List[int]]] = None,
    vis_kp_index: bool = False,
) -> Union[None, torch.Tensor]:
    """Render SMPL or SMPL-X mesh or silhouette into differentiable tensors,
    and export video or images.

    Args:
        # smpl parameters:
        poses (Union[torch.Tensor, np.ndarray, dict]):
            1). `tensor` or `array` and ndim is 2, shape should be
                (frame, 72).
            2). `tensor` or `array` and ndim is 3, shape should be
                (frame, num_person, 72/165). num_person equals 1 means
                single-person.
                Rendering predicted multi-person should feed together with
                multi-person weakperspective cameras. meshes would be computed
                and use an identity intrinsic matrix.
            3). `dict`, standard dict format defined in smplx.body_models.
                will be treated as single-person.
            Lower priority than `verts`.
            Defaults to None.
        betas (Optional[Union[torch.Tensor, np.ndarray]], optional):
            1). ndim is 2, shape should be (frame, 10).
            2). ndim is 3, shape should be (frame, num_person, 10). num_person
                equals 1 means single-person. If poses are multi-person, betas
                should be set to the same person number.
            None will use default betas.
            Defaults to None.
        transl (Optional[Union[torch.Tensor, np.ndarray]], optional):
            translations of smpl(x).
            1). ndim is 2, shape should be (frame, 3).
            2). ndim is 3, shape should be (frame, num_person, 3). num_person
                equals 1 means single-person. If poses are multi-person,
                transl should be set to the same person number.
            Defaults to None.
        verts (Optional[Union[torch.Tensor, np.ndarray]], optional):
            1). ndim is 3, shape should be (frame, num_verts, 3).
            2). ndim is 4, shape should be (frame, num_person, num_verts, 3).
                num_person equals 1 means single-person.
            Higher priority over `poses` & `betas` & `transl`.
            Defaults to None.
        model_type (Literal[, optional): choose in 'smpl' or 'smplx'.
            Defaults to 'smpl'.
        gender (Literal[, optional): chose in ['male', 'female', 'neutral'].
            Defaults to 'neutral'.
        body_model_dir (str, optional): Directory of npz or pkl path.
            Lower priority than `body_model`.
            Defaults to None.
        body_model (nn.Module, optional): body_model created from smplx.create.
            Higher priority than `body_model_dir`. Should not both be None.
            Defaults to None.

        # camera parameters:
        K (Optional[Union[torch.Tensor, np.ndarray]], optional):
            shape should be (frame, 4, 4) or (frame, 3, 3), frame could be 1.
            if (4, 4) or (3, 3), dim 0 will be added automatically.
            Will be default `FovPerspectiveCameras` intrinsic if None.
            Lower priority than `pred_cam`.
        R (Optional[Union[torch.Tensor, np.ndarray]], optional):
            shape should be (frame, 3, 3), If f equals 1, camera will have
            identical rotation.
            If `K` and `pred_cam` is None, will be generated by `look_at_view`.
            If have `K` or `pred_cam` and `R` is None, will be generated by
            `convert_cameras`.
            Defaults to None.
        T (Optional[Union[torch.Tensor, np.ndarray]], optional):
            shape should be (frame, 3). If f equals 1, camera will have
            identical translation.
            If `K` and `pred_cam` is None, will be generated by `look_at_view`.
            If have `K` or `pred_cam` and `T` is None, will be generated by
            `convert_cameras`.
            Defaults to None.
        pred_cam (Optional[Union[torch.Tensor, np.ndarray]], optional):
            shape should be (frame, 4) or (frame, num_person, 4). If f equals
            1, will be repeated to num_frame. num_person should be 1 if single
            person. Usually for HMR, VIBE predicted cameras.
            Higher priority than `K` & `R` & `T`.
            Defaults to None.
        in_ndc (bool, optional): . Defaults to True.
        convention (str, optional): If want to  use an existing convention,
            choose in ['opengl', 'opencv', 'pytorch3d', 'pyrender', 'open3d',
            'maya', 'blender', 'unity'].
            If want to use a new convention, define your convention in
            (CAMERA_CONVENTION_FACTORY)[mmhuman3d/core/conventions/cameras/
            __init__.py] by the order of right, front and up.
            Defaults to 'pytorch3d'.
        projection (Literal[, optional): projection mode of camers. Choose in
            ['orthographics, fovperspective', 'perspective', 'weakperspective',
            'fovorthographics']
            Defaults to 'perspective'.
        orbit_speed (float, optional): orbit speed for viewing when no `K`
            provided. `float` for only azim speed and Tuple for `azim` and
            `elev`.

        # render choice parameters:
        render_choice (Literal[, optional):
            choose in ['lq', 'mq', 'hq', 'silhouette', 'part_silhouette'].
            lq, mq, hq would output (frame, h, w, 4) tensor.
            lq means low quality, mq means medium quality,
            hq means high quality.
            silhouette would output (frame, h, w) binary tensor.
            part_silhouette would output (frame, h, w, n_class) tensor.
            n_class is the body segmentation classes.
            Defaults to 'mq'.
        palette (Union[List[str], str, np.ndarray], optional):
            color theme str or list of color str or `array`.
            1). If use str to represent the color,
                should choose in ['segmentation', 'blue',
                'red', 'white', 'black', 'green', 'yellow', 'random']
                If choose 'segmentation', will get a color for each part.
            2). If you have multi-person, better give a list of str or all
                will be in the same color.
            3). If you want to define your specific color, use an `array`
                of shape (3,) for singe person and (N, 3) for multiple person.
                If (3,) for multiple person, all will be in the same color.
                Your `array` should be in range [0, 255] for 8 bit color.
            Defaults to 'white'.
        resolution (Union[Iterable[int], int], optional):
            1). If iterable, should be (height, width) of output images.
            2). If int, would be taken as (resolution, resolution).
            Defaults to (1024, 1024).
        start (int, optional): start frame index. Defaults to 0.
        end (int, optional): end frame index. Defaults to -1.
        alpha (float, optional): Transparency of the mesh.
            Range in [0.0, 1.0]
            Defaults to 1.0.
        no_grad (bool, optional): Set to True if do not need differentiable
            render. Defaults to False.
        batch_size (int, optional):  Batch size for render.
            Related to your gpu memory.
            Defaults to 20.

        # TODO
        DDP (bool, optional): whether use distributeddataparallel.
            Defaults to False.

        # file io parameters:
        return_tensor (bool, optional): Whether return the result tensors.
            Defaults to False, will return None.
        output_path (str, optional): output video or gif or image folder.
            Defaults to None, pass export procedure.

        # background frames, priority: image_array > frame_list > origin_frames
        origin_frames (Optional[str], optional): origin brackground frame path,
            could be `.mp4`, `.gif`(will be sliced into a folder) or an image
            folder.
            Defaults to None.
        frame_list (Optional[List[str]], optional): list of origin brackground
            frame paths, element in list each should be a image path like
            `*.jpg` or `*.png`.
            Use this when your file names is hard to sort or you only want to
            render a small number frames.
            Defaults to None.
        image_array: (Optional[Union[np.ndarray, torch.Tensor]], optional):
            origin brackground frame `tensor` or `array`, use this when you
            want your frames in memory as array or tensor.
        overwrite (bool, optional): whether overwriting the existing files.
            Defaults to False.
        obj_path (bool, optional): the directory path to store the `.obj`
            files. Defaults to None.
        read_frames_batch (bool, optional): [description]. Defaults to False.

        # visualize keypoints
        plot_kps (bool, optional): [description]. Defaults to False.
        kp3d (Optional[Union[np.ndarray, torch.Tensor]], optional):
            the keypoints of any convention, should pass `mask` if have any
            none-sense points. Shape should be (frame, )
            Defaults to None.
        smpl_joints (Optional[Union[np.ndarray, torch.Tensor]], optional):
            smpl joints output by body_model, required if pass `verts`.
            Defaults to None.
        mask (Optional[Union[np.ndarray, List[int]]], optional):
            Mask of keypoints existence. Defaults to None.

    Returns:
        Union[None, torch.Tensor]: return the rendered image tensors or None.
    """
    # TODO, use DDP
    # initialize the gpu device
    gpu_count = int(torch.cuda.device_count())
    gpu_list = [i for i in range(gpu_count)]
    gpu_str = ','.join(list(map(lambda x: str(x), gpu_list)))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if verts is None and poses is None:
        raise ValueError('Please input valid poses or verts.')
    elif (verts is not None) and (poses is not None):
        warnings.warn('Redundant input, will take verts and ignore poses & '
                      'betas & transl.')
        poses = None
        transl = None
        betas = None

    if isinstance(verts, np.ndarray):
        verts = torch.Tensor(verts)
        num_frame = verts.shape[0]
    elif isinstance(verts, torch.Tensor):
        num_frame = verts.shape[0]

    if isinstance(poses, np.ndarray):
        poses = torch.Tensor(poses)
        num_frame = poses.shape[0]
    elif isinstance(poses, torch.Tensor):
        num_frame = poses.shape[0]
    elif isinstance(poses, dict):
        for k, v in poses.items():
            if isinstance(v, np.ndarray):
                poses[k] = torch.tensor(v)
        num_frame = poses['body_pose'].shape[0]

    if isinstance(image_array, np.ndarray):
        image_array = torch.Tensor(image_array)

    if isinstance(betas, np.ndarray):
        betas = torch.Tensor(betas)

    if betas is not None:
        if betas.shape[0] != num_frame:
            times = num_frame // betas.shape[0]
            if betas.ndim == 2:
                betas = betas.repeat(times, 1)[:num_frame]
            elif betas.ndim == 3:
                betas = betas.repeat(times, 1, 1)[:num_frame]
            print(f'betas will be repeated by dim 0 for {times} times.')
    if isinstance(transl, np.ndarray):
        transl = torch.Tensor(transl)

    # multi person check
    if isinstance(poses, torch.Tensor):
        if poses.ndim == 4 and poses.shape[1] > 1:
            if betas is not None:
                betas = betas.view(num_frame, -1, 10)
                print(f'betas will be repeated by dim 0 for {times} times.')
                if betas.shape[1] == 1:
                    warnings.warn(
                        'Only one betas for multi-person, will all be the '
                        'same body shape.')
            else:
                warnings.warn('None betas for multi-person, will all be the '
                              'default body shape.')

            if transl is not None:
                transl = transl.view(poses.shape[0], -1, 10)
                if transl.shape[1] == 1:
                    warnings.warn(
                        'Only one transl for multi-person, will all be the '
                        'same translation.')
            else:
                warnings.warn('None transl for multi-person, will all be the '
                              'default translation.')
    if isinstance(pred_cam, np.ndarray):
        pred_cam = torch.Tensor(pred_cam)

    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    elif isinstance(resolution, list):
        resolution = tuple(resolution)

    # init smpl(x) body model
    if model_type not in ['smpl', 'smplx']:
        raise ValueError(
            f'Do not support {model_type}, please choose in `smpl` or `smplx.')
    if body_model is None:
        if body_model_dir is not None:
            if not (Path(body_model_dir).is_dir()
                    or Path(body_model_dir).is_file()):
                raise FileNotFoundError('Wrong body_model_dir.'
                                        ' File or directory does not exist.')
            body_model = get_body_model(
                model_path=body_model_dir,
                model_type=model_type,
                gender=gender,
                use_pca=use_pca)
        else:
            raise ValueError('Please input body_model or body_model_dir.')
    else:
        if body_model_dir is not None:
            warnings.warn('Redundant input, will take body_model directly'
                          'and ignore body_model_dir.')

    vertices, faces, joints, num_frame, num_person, start, end = _prepare_mesh(
        poses, betas, transl, verts, start, end, body_model)
    if not plot_kps:
        joints = None
        if kp3d is not None:
            warnings.warn('`plot_kps` is False, `kp3d` will be set as None.')
            kp3d = None
        if smpl_joints is not None:
            warnings.warn(
                '`plot_kps` is False, `smpl_joints` will be set as None.')
            smpl_joints = None

    if num_frame > 500:
        read_frames_batch = True

    frames_folder = None
    remove_folder = False

    if image_array is not None:
        frame_list = None
        origin_frames = None
        image_array = image_array[start:end]
    # check the output path and get the image_array
    if output_path is not None:
        prepare_output_path(
            output_path=output_path,
            allowed_suffix=['.mp4', 'gif', ''],
            tag='output video',
            path_type='auto',
            overwrite=overwrite)
        if image_array is None:
            # choose in frame_list or origin_frames
            if frame_list is None and origin_frames is None:
                print(
                    'No background provided, will use pure white background.')
            elif frame_list is not None and origin_frames is not None:
                warnings.warn('Redundant input, will only use frame_list.')
                origin_frames = None

            # read the origin frames as array if any.
            if origin_frames is not None and frame_list is None:
                check_input_path(
                    input_path=origin_frames,
                    allowed_suffix=['.mp4', '.gif', ''],
                    tag='origin frames',
                    path_type='auto')
                if Path(origin_frames).is_file():
                    if read_frames_batch:
                        frames_folder = osp.join(
                            Path(output_path).parent,
                            Path(output_path).name + '_input_temp')
                        os.makedirs(frames_folder, exist_ok=True)
                        video_to_images(
                            origin_frames, frames_folder, start=start, end=end)
                        remove_folder = True
                    else:
                        remove_folder = False
                        frames_folder = None
                        image_array = video_to_array(
                            origin_frames, start=start, end=end)
                else:
                    if read_frames_batch:
                        frames_folder = origin_frames
                        remove_folder = False
                        image_array = None
                    else:
                        image_array = images_to_array(
                            origin_frames,
                            img_format=img_format,
                            start=start,
                            end=end)
                        remove_folder = False
                        frames_folder = origin_frames
            elif frame_list is not None and origin_frames is None:
                frames_folder = osp.join(
                    Path(output_path).parent,
                    Path(output_path).name + '_input_temp')
                os.makedirs(frames_folder, exist_ok=True)
                for frame_idx, frame_path in enumerate(frame_list):
                    if check_path_suffix(
                            path_str=frame_path,
                            allowed_suffix=['.jpg', '.png', '.jpeg']):
                        shutil.copy(
                            frame_path,
                            os.path.join(frames_folder,
                                         '%06d.png' % frame_idx))
                        img_format = '%06d.png'
                if not read_frames_batch:
                    image_array = images_to_array(
                        frames_folder,
                        img_format=img_format,
                        remove_raw_files=True)
                    frames_folder = None
                    remove_folder = False
                else:
                    image_array = None
                    remove_folder = True

    vertices = vertices.view(num_frame, num_person, -1, 3)

    if isinstance(kp3d, np.ndarray):
        kp3d = torch.Tensor(kp3d)
    if isinstance(smpl_joints, np.ndarray):
        smpl_joints = torch.Tensor(smpl_joints)

    if smpl_joints is not None:
        if mask is not None:
            map_index = np.where(np.array(mask) != 0)[0]
            smpl_joints = smpl_joints[map_index.tolist()]
        smpl_joints = smpl_joints[start:end + 1]
        smpl_joints = smpl_joints.view(num_frame, -1, 3)
    joints = joints if joints is not None else smpl_joints

    if kp3d is not None:
        kp3d = kp3d[start:end + 1]
        kp3d = kp3d.view(num_frame, -1, 3)

    # prepare render_param_dict
    if render_choice not in [
            'hq', 'mq', 'lq', 'silhouette', 'part_silhouette', 'depth',
            'pointcloud'
    ]:
        raise ValueError('Please choose the right render_choice.')
    if render_choice not in ['depth', 'pointcloud']:
        render_param_dict = copy.deepcopy(
            RENDER_CONFIGS[render_choice.lower()])
    else:
        render_param_dict = copy.deepcopy(RENDER_CONFIGS['lq'])
    # body part colorful visualization should use flat shader to be shaper.
    if isinstance(palette, str):
        if (palette == 'segmentation') and ('silhouette'
                                            not in render_choice.lower()):
            render_param_dict['shader']['shader_type'] = 'flat'
        palette = [palette] * num_person
    elif isinstance(palette, np.ndarray):
        palette = palette.view(-1, 3)
        if palette.shape[0] != num_person:
            _times = num_person // palette.shape[0]
            palette = palette.repeat(_times, 0)[:num_person]
            if palette.shape[0] == 1:
                print(f'Same color for all the {num_person} people')
            else:
                print('Repeat palette for multi-person.')

    if not len(palette) == num_person:
        raise ValueError('Please give the right number of palette.')

    # slice the input cameras and check the type and shape
    if pred_cam is not None:
        assert projection == 'weakperspective', 'Providing `pred_cam` should '
        'set `projection` as `weakperspective` at the same time.'
        r = resolution[1] / resolution[0]
        pred_cam = pred_cam[start:end + 1]
        pred_cam = pred_cam.view(num_frame, num_person, 4)
        # if num_person > 1:
        sx, sy, tx, ty = torch.unbind(pred_cam, -1)

        vertices[..., 0] += tx.view(num_frame, num_person, 1)
        vertices[..., 1] += ty.view(num_frame, num_person, 1)
        vertices[..., 0] *= sx.view(num_frame, num_person, 1)
        vertices[..., 1] *= sy.view(num_frame, num_person, 1)
        pred_cam = torch.tensor([1.0, 1.0, 0.0,
                                 0.0]).view(1, 4).repeat(num_frame, 1)
        K, R, T = WeakPerspectiveCameras.convert_pred_cam_to_matrix(
            pred_cam=pred_cam,
            znear=torch.min(vertices[..., 2] - 1),
            aspect_ratio=r)
    # pred_cam and K are None, use look_at_view
    if K is None:
        projection = 'fovperspective'
        K, R, T = compute_orbit_cameras(
            at=(torch.mean(vertices.view(-1, 3), 0)),
            orbit_speed=orbit_speed,
            batch_size=num_frame,
            convention=convention)
        convention = 'pytorch3d'
    if isinstance(R, np.ndarray):
        R = torch.Tensor(R).view(-1, 3, 3)
    elif isinstance(R, torch.Tensor):
        R = R.view(-1, 3, 3)
    else:
        R = None

    if R is not None:
        if len(R) > num_frame:
            R = R[start:end + 1]

    if isinstance(T, np.ndarray):
        T = torch.Tensor(T).view(-1, 3)
    elif isinstance(T, torch.Tensor):
        T = T.view(-1, 3)
    else:
        T = None

    if T is not None:
        if len(T) > num_frame:
            T = T[start:end + 1]

    if isinstance(K, np.ndarray):
        K = torch.Tensor(K).view(-1, K.shape[-2], K.shape[-1])
    elif isinstance(K, torch.Tensor):
        K = K.view(-1, K.shape[-2], K.shape[-1])
    else:
        K = None

    if K is not None:
        if len(K) > num_frame:
            K = K[start:end + 1]

    assert projection in [
        'perspective', 'weakperspective', 'orthographics', 'fovorthographics',
        'fovperspective'
    ]
    if projection in ['fovperspective', 'perspective']:
        is_perspective = True
    elif projection in [
            'fovorthographics', 'weakperspective', 'orthographics'
    ]:
        is_perspective = False

    K, R, T = convert_cameras(
        convention_dst='pytorch3d',
        K=K,
        R=R,
        T=T,
        is_perspective=is_perspective,
        convention_src=convention,
        resolution_src=resolution,
        in_ndc_src=in_ndc)

    # initialize the renderer.
    renderer = SMPLRenderer(
        resolution=resolution,
        faces=faces,
        device=device,
        obj_path=obj_path,
        output_path=output_path,
        palette=palette,
        return_tensor=return_tensor,
        alpha=alpha,
        model_type=model_type,
        img_format=img_format,
        render_choice=render_choice,
        projection=projection,
        frames_folder=frames_folder,
        plot_kps=plot_kps,
        vis_kp_index=vis_kp_index,
        **render_param_dict)

    renderer = renderer.to(device)

    # prepare the render data.
    render_dataset = RenderDataset(
        **{
            'images': image_array,
            'vertices': vertices,
            'K': K,
            'R': R,
            'T': T,
            'joints': joints,
            'joints_gt': kp3d,
        })

    RenderLoader = DataLoader(
        dataset=render_dataset, batch_size=batch_size, shuffle=False)

    # TODO: complete DDP for multi-gpu rendering.
    if len(gpu_list) > 1:
        renderer = nn.parallel.DataParallel(renderer, device_ids=gpu_list)

    # start rendering. no grad if non-differentiable render.
    # return None if return_tensor is False.
    results = []
    for data in tqdm(RenderLoader):
        if no_grad:
            with torch.no_grad():
                result = renderer(**data)
                torch.cuda.empty_cache()
                results.append(result)
        else:
            result = renderer(**data)
            results.append(result)
    renderer.export()

    if remove_folder:
        if Path(frames_folder).is_dir():
            shutil.rmtree(frames_folder)

    if return_tensor:
        results = torch.cat(results, 0)
        return results
    else:
        return None


def neural_render_smpl(
        poses,
        betas=None,
        transl=None,
        verts=None,
        render_choice='silhouette',
        projection='weakperspective',
        convention='opencv',
        pred_cam=None,
        K=None,
        R=None,
        T=None,
        body_model_dir=None,
        body_model=None,
        model_type='smpl',
        gender='neutral',
        resolution=(1024, 1024),
        batch_size=10,
):
    """Differentiable render silhouette of SMPL(X) mesh.

    You could specify these following parameters:
        poses,  betas, transl, verts, pred_cam, body_model_dir,
        model_type, resolution, start, end, gender, batch_size.
    Returns:
        torch.Tensor: Would be shape of (frame, H, W, 1)
    """
    assert pred_cam is not None, '`pred_cam` is required.'
    func = partial(
        render_smpl,
        return_tensor=True,
        obj_path=None,
        origin_frames=None,
        frame_list=None,
        image_array=None,
        output_path=None,
        no_grad=False,
        in_ndc=True,
        start=0,
        end=-1,
        palette='white',
        overwrite=False)

    return func(
        poses=poses,
        betas=betas,
        transl=transl,
        render_choice=render_choice,
        verts=verts,
        projection=projection,
        convention=convention,
        pred_cam=pred_cam,
        K=K,
        R=R,
        T=T,
        body_model_dir=body_model_dir,
        body_model=body_model,
        model_type=model_type,
        gender=gender,
        resolution=resolution,
        batch_size=batch_size)


def visualize_smpl_opencv(poses=None,
                          betas=None,
                          verts=None,
                          transl=None,
                          model_type='smpl',
                          gender='neutral',
                          body_model_dir=None,
                          body_model=None,
                          palette='random',
                          batch_size=10,
                          K=None,
                          R=None,
                          T=None,
                          output_path='sample.mp4',
                          resolution=None,
                          render_choice='hq',
                          origin_frames=None,
                          overwrite=True,
                          start=0,
                          end=-1,
                          alpha=1.0,
                          kp3d=None,
                          vis_kp_index=False,
                          mask=None) -> None:
    """Visualize a smpl mesh which has opencv calibration matrix defined in
    screen.

    Require K, R, T and resolution.
    """
    if render_choice == 'pointcloud':
        plot_kps = True
    else:
        plot_kps = False
    if render_choice.lower() not in ['hq', 'mq', 'lq', 'depth', 'pointcloud']:
        raise ValueError(
            "Please choose render_choice in ['hq', 'mq', 'lq', 'depth',"
            " 'pointcloud'].")
    assert K is not None, '`K` is required.'
    assert R is not None, '`R` is required.'
    assert T is not None, '`T` is required.'
    assert resolution is not None, '`resolution`(h, w) is required.'
    func = partial(
        render_smpl,
        projection='perspective',
        convention='opencv',
        pred_cam=None,
        in_ndc=False,
        return_tensor=False,
        no_grad=True,
        obj_path=None)
    func(
        poses=poses,
        betas=betas,
        verts=verts,
        transl=transl,
        K=K,
        R=R,
        T=T,
        batch_size=batch_size,
        palette=palette,
        body_model_dir=body_model_dir,
        body_model=body_model,
        model_type=model_type,
        render_choice=render_choice,
        output_path=output_path,
        overwrite=overwrite,
        resolution=resolution,
        gender=gender,
        origin_frames=origin_frames,
        start=start,
        end=end,
        alpha=alpha,
        plot_kps=plot_kps,
        kp3d=kp3d,
        vis_kp_index=vis_kp_index,
        mask=mask)


def visualize_smpl_pred(poses=None,
                        betas=None,
                        verts=None,
                        transl=None,
                        model_type='smpl',
                        gender='neutral',
                        body_model_dir=None,
                        body_model=None,
                        palette='random',
                        img_format='%06d.png',
                        batch_size=10,
                        pred_cam=None,
                        output_path='sample.mp4',
                        resolution=(1024, 1024),
                        render_choice='hq',
                        origin_frames=None,
                        overwrite=True,
                        alpha=1.0,
                        start=0,
                        end=-1,
                        read_frames_batch=False,
                        kp3d=None,
                        vis_kp_index=False,
                        obj_path=None,
                        mask=None) -> None:
    """Simpliest way to visualize pred smpl with orign frames and predicted
    cameras. `render_choice` should be in 'hq', 'mq', 'lq'.

    Returns:
        None
    """
    if render_choice == 'pointcloud':
        plot_kps = True
    else:
        plot_kps = False
    if render_choice.lower() not in ['hq', 'mq', 'lq', 'depth', 'pointcloud']:
        raise ValueError(
            "Please choose render_choice in ['hq', 'mq', 'lq', 'depth',"
            " 'pointcloud'].")
    assert pred_cam is not None, '`pred_cam` is required.'
    func = partial(
        render_smpl,
        K=None,
        R=None,
        T=None,
        projection='weakperspective',
        convention='opencv',
        in_ndc=True,
        return_tensor=False,
        no_grad=True,
    )
    func(
        poses=poses,
        betas=betas,
        verts=verts,
        transl=transl,
        pred_cam=pred_cam,
        batch_size=batch_size,
        palette=palette,
        body_model_dir=body_model_dir,
        body_model=body_model,
        model_type=model_type,
        render_choice=render_choice,
        output_path=output_path,
        overwrite=overwrite,
        resolution=resolution,
        gender=gender,
        origin_frames=origin_frames,
        start=start,
        end=end,
        obj_path=obj_path,
        img_format=img_format,
        alpha=alpha,
        read_frames_batch=read_frames_batch,
        plot_kps=plot_kps,
        kp3d=kp3d,
        vis_kp_index=vis_kp_index,
        mask=mask)


def visualize_T_pose(poses,
                     body_model_dir=None,
                     body_model=None,
                     output_path='sample.mp4',
                     orbit_speed=0.0,
                     model_type='smpl',
                     render_choice='hq',
                     palette='white',
                     overwrite=False,
                     kp3d=None,
                     resolution=(1024, 1024),
                     vis_kp_index=False,
                     batch_size=10) -> None:
    """Simpliest way to visualize a sequence of T pose or smpl pose without
    `global_orient`. `render_choice` should be in 'hq', 'mq', 'lq'.

    Returns:
        None
    """
    assert poses is not None, '`poses` is required.'
    if render_choice.lower() not in ['hq', 'mq', 'lq', 'depth', 'pointcloud']:
        raise ValueError(
            "Please choose render_choice in ['hq', 'mq', 'lq', 'depth',"
            " 'pointcloud'].")
    if render_choice == 'pointcloud':
        plot_kps = True
    else:
        plot_kps = False
    func = partial(
        render_smpl,
        betas=None,
        transl=None,
        verts=None,
        convention='pytorch3d',
        projection='fovperspective',
        K=None,
        R=None,
        T=None,
        return_tensor=False,
        no_grad=True,
        obj_path=None,
        origin_frames=None,
        gender='neutral')
    func(
        poses=poses,
        body_model_dir=body_model_dir,
        body_model=body_model,
        model_type=model_type,
        orbit_speed=orbit_speed,
        render_choice=render_choice,
        palette=palette,
        output_path=output_path,
        overwrite=overwrite,
        vis_kp_index=vis_kp_index,
        plot_kps=plot_kps,
        kp3d=kp3d,
        resolution=resolution,
        batch_size=batch_size)


def visualize_smpl_pose(
    poses=None,
    verts=None,
    body_model_dir=None,
    body_model=None,
    output_path='sample.mp4',
    orbit_speed=0.0,
    model_type='smpl',
    render_choice='hq',
    overwrite=False,
    batch_size=10,
    palette='white',
    resolution=(1024, 1024),
    start=0,
    end=-1,
    kp3d=None,
    vis_kp_index=False,
    mask=None,
) -> None:
    """Simpliest way to visualize a sequence of smpl pose. Cameras will focus
    on the center of smpl mesh. `orbit speed` is recomended. `render_choice`
    should be in 'hq', 'mq', 'lq'.

    Returns:
        None
    """
    assert poses is not None or verts is not None, \
        '`poses` or `verts` is required.'
    if render_choice.lower() not in ['hq', 'mq', 'lq', 'depth', 'pointcloud']:
        raise ValueError(
            "Please choose render_choice in ['hq', 'mq', 'lq', 'depth',"
            " 'pointcloud'].")
    if render_choice == 'pointcloud':
        plot_kps = True
    else:
        plot_kps = False
    func = partial(
        render_smpl,
        betas=None,
        transl=None,
        convention='opencv',
        projection='fovperspective',
        K=None,
        R=None,
        T=None,
        in_ndc=True,
        return_tensor=False,
        no_grad=True,
        obj_path=None,
        origin_frames=None,
        gender='neutral')
    func(
        poses=poses,
        verts=verts,
        batch_size=batch_size,
        palette=palette,
        body_model_dir=body_model_dir,
        body_model=body_model,
        model_type=model_type,
        orbit_speed=orbit_speed,
        render_choice=render_choice,
        output_path=output_path,
        overwrite=overwrite,
        resolution=resolution,
        start=start,
        end=end,
        plot_kps=plot_kps,
        kp3d=kp3d,
        vis_kp_index=vis_kp_index,
        mask=mask)
