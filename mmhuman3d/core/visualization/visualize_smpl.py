import copy
import glob
import os
import os.path as osp
import shutil
import warnings
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mmcv
import numpy as np
import torch
import torch.nn as nn
from colormap import Color
from torch.utils.data import DataLoader
from tqdm import tqdm

import mmhuman3d
from mmhuman3d.core.cameras import (
    WeakPerspectiveCameras,
    compute_orbit_cameras,
)
from mmhuman3d.core.conventions.cameras import convert_cameras
from mmhuman3d.core.conventions.segmentation import body_segmentation
from mmhuman3d.models.builder import build_body_model
from mmhuman3d.utils import (
    check_input_path,
    check_path_suffix,
    convert_bbox_to_intrinsic,
    convert_crop_cam_to_orig_img,
    convert_kp2d_to_bbox,
    get_default_hmr_intrinsic,
    get_different_colors,
    images_to_array,
    prepare_output_path,
    save_meshes_as_plys,
    vid_info_reader,
    video_to_array,
    video_to_images,
)
from .renderer import RenderDataset, SMPLRenderer

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _prepare_background(image_array, frame_list, origin_frames, output_path,
                        start, end, img_format, overwrite, num_frames,
                        read_frames_batch):
    """Compare among `image_array`, `frame_list` and `origin_frames` and decide
    whether to save the temp background images."""
    if num_frames > 300:
        read_frames_batch = True

    frames_folder = None
    remove_folder = False

    if isinstance(image_array, np.ndarray):

        image_array = torch.Tensor(image_array)

    if image_array is not None:
        if image_array.ndim == 3:
            image_array = image_array[None]
        if image_array.shape[0] == 1:
            image_array = image_array.repeat(num_frames, 1, 1, 1)
        image_array
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
            # if all None, will use pure white background
            if frame_list is None and origin_frames is None:
                print(
                    'No background provided, will use pure white background.')
            elif frame_list is not None and origin_frames is not None:
                warnings.warn('Redundant input, will only use frame_list.')
                origin_frames = None

            # read the origin frames as array if any.
            if frame_list is None and origin_frames is not None:
                check_input_path(
                    input_path=origin_frames,
                    allowed_suffix=['.mp4', '.gif', ''],
                    tag='origin frames',
                    path_type='auto')
                # if origin_frames is a video, write it as a folder of images
                # if read_frames_batch is True, else read directly as an array.
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
                # if origin_frames is a folder, write it as a folder of images
                # read the folder as an array if read_frames_batch is True
                # else return frames_folder for reading during rendering.
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
            # if frame_list is not None, move the images into a folder
            # read the folder as an array if read_frames_batch is True
            # else return frames_folder for reading during rendering.
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
    return image_array, remove_folder, frames_folder


def _prepare_body_model(model_type, body_model, body_model_config):
    """Prepare `body_model` from `body_model_config` or existing
    `body_model`."""
    if model_type not in ['smpl', 'smplx']:
        raise ValueError(
            f'Do not support {model_type}, please choose in `smpl` or `smplx.')
    if body_model is None:
        if body_model_config is not None:
            body_model_config = copy.deepcopy(body_model_config)
            model_path = body_model_config.get('model_path', None)

            model_type = body_model_config.get('type', model_type)
            body_model_config.update(type=model_type.lower())

            if model_path and osp.isdir(model_path):
                model_path = osp.join(model_path, model_type)
                body_model_config.update(model_path=model_path)
                body_model = build_body_model(body_model_config)
                assert os.path.isdir(model_path)
            else:
                raise FileNotFoundError('Wrong model_path.'
                                        ' File or directory does not exist.')
        else:
            raise ValueError('Please input body_model_config.')
    else:
        if body_model_config is not None:
            warnings.warn('Redundant input, will take body_model directly'
                          'and ignore body_model_config.')
    return body_model


def _prepare_input_pose(verts, poses, betas, transl):
    """Prepare input pose data as tensor and ensure correct temporal slice."""
    if verts is None and poses is None:
        raise ValueError('Please input valid poses or verts.')
    elif (verts is not None) and (poses is not None):
        warnings.warn('Redundant input, will take verts and ignore poses & '
                      'betas & transl.')
        poses = None
        transl = None
        betas = None
    elif isinstance(poses, dict):
        transl = poses.get('transl', transl)
        betas = poses.get('betas', betas)

    if isinstance(verts, np.ndarray):
        verts = torch.Tensor(verts)
        num_frames = verts.shape[0]
    elif isinstance(verts, torch.Tensor):
        num_frames = verts.shape[0]

    if isinstance(poses, np.ndarray):
        poses = torch.Tensor(poses)
        num_frames = poses.shape[0]
    elif isinstance(poses, torch.Tensor):
        num_frames = poses.shape[0]
    elif isinstance(poses, dict):
        for k, v in poses.items():
            if isinstance(v, np.ndarray):
                poses[k] = torch.tensor(v)
        num_frames = poses['body_pose'].shape[0]

    if isinstance(betas, np.ndarray):
        betas = torch.Tensor(betas)

    if betas is not None:
        if betas.shape[0] != num_frames:
            times = num_frames // betas.shape[0]
            if betas.ndim == 2:
                betas = betas.repeat(times, 1)[:num_frames]
            elif betas.ndim == 3:
                betas = betas.repeat(times, 1, 1)[:num_frames]
            print(f'betas will be repeated by dim 0 for {times} times.')
    if isinstance(transl, np.ndarray):
        transl = torch.Tensor(transl)

    return verts, poses, betas, transl


def _prepare_mesh(poses, betas, transl, verts, start, end, body_model):
    """Prepare the mesh info for rendering."""
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
            num_frames = poses['body_pose'].shape[0]
            _, num_person, _ = poses['body_pose'].view(
                num_frames, -1, NUM_BODY_JOINTS * 3).shape

            full_pose = body_model.dict2tensor(poses)
            full_pose = full_pose[start:end]

        elif isinstance(poses, torch.Tensor):
            if poses.shape[-1] != NUM_DIM:
                raise ValueError(
                    f'Please make sure your poses is {NUM_DIM} dims in'
                    f'the last axis. Your input shape: {poses.shape}')
            poses = poses.view(poses.shape[0], -1, (NUM_JOINTS + 1) * 3)
            num_frames, num_person, _ = poses.shape
            full_pose = poses[start:end]
        else:
            raise ValueError('Wrong pose type, should be `dict` or `tensor`.')

        # multi person check
        if num_person > 1:
            if betas is not None:
                betas = betas.view(num_frames, -1, 10)

                if betas.shape[1] == 1:
                    betas = betas.repeat(1, num_person, 1)
                    warnings.warn(
                        'Only one betas for multi-person, will all be the '
                        'same body shape.')
                elif betas.shape[1] > num_person:
                    betas = betas[:, :num_person]
                    warnings.warn(
                        f'Betas shape exceed, will be sliced as {betas.shape}.'
                    )
                elif betas.shape[1] == num_person:
                    pass
                else:
                    raise ValueError(
                        f'Odd betas shape: {betas.shape}, inconsistent'
                        f'with poses in num_person: {poses.shape}.')
            else:
                warnings.warn('None betas for multi-person, will all be the '
                              'default body shape.')

            if transl is not None:
                transl = transl.view(poses.shape[0], -1, 3)
                if transl.shape[1] == 1:
                    transl = transl.repeat(1, num_person, 1)
                    warnings.warn(
                        'Only one transl for multi-person, will all be the '
                        'same translation.')
                elif transl.shape[1] > num_person:
                    transl = transl[:, :num_person]
                    warnings.warn(f'Transl shape exceed, will be sliced as'
                                  f'{transl.shape}.')
                elif transl.shape[1] == num_person:
                    pass
                else:
                    raise ValueError(
                        f'Odd transl shape: {transl.shape}, inconsistent'
                        f'with poses in num_person: {poses.shape}.')
            else:
                warnings.warn('None transl for multi-person, will all be the '
                              'default translation.')

        # slice the input poses, betas, and transl.
        betas = betas[start:end] if betas is not None else None
        transl = transl[start:end] if transl is not None else None
        pose_dict = body_model.tensor2dict(
            full_pose=full_pose, betas=betas, transl=transl)

        # get new num_frames
        num_frames = full_pose.shape[0]

        model_output = body_model(**pose_dict)
        vertices = model_output['vertices']
        faces = body_model.faces_tensor
        joints = model_output['joints']

    elif verts is not None:
        if isinstance(verts, np.ndarray):
            verts = torch.Tensor(verts)

        pose_dict = body_model.tensor2dict(
            torch.zeros(1, (NUM_JOINTS + 1) * 3))

        if verts.ndim == 3:
            joints = torch.einsum('bik,ji->bjk',
                                  [verts, body_model.J_regressor])
        elif verts.ndim == 4:
            joints = torch.einsum('fpik,ji->fpjk',
                                  [verts, body_model.J_regressor])
        model_output = body_model(**pose_dict)
        num_verts = body_model.NUM_VERTS
        assert verts.shape[-2] == num_verts, 'Wrong input verts shape.'
        faces = body_model.faces_tensor
        num_frames = verts.shape[0]
        verts = verts[start:end]
        num_frames = verts.shape[0]
        vertices = verts.view(num_frames, -1, num_verts, 3)
        num_joints = joints.shape[-2]
        joints = joints.view(num_frames, -1, num_joints, 3)
        num_person = vertices.shape[1]
    else:
        raise ValueError('Poses and verts are all None.')
    return vertices, faces, joints, num_frames, num_person


def _prepare_colors(palette, render_choice, num_person, num_verts, model_type):
    """Prepare the `color` as a tensor of shape (num_person, num_verts, 3)
    according to `palette`.

    This is to make the identity in video clear.
    """
    if not len(palette) == num_person:
        raise ValueError('Please give the right number of palette.')
    body_segger = body_segmentation(model_type)

    if render_choice == 'silhouette':
        colors = torch.ones(num_person, num_verts, 3)
    elif render_choice == 'part_silhouette':
        colors = torch.zeros(num_person, num_verts, 3)
        for i, k in enumerate(body_segger.keys()):
            colors[:, body_segger[k]] = i + 1
    else:
        if isinstance(palette, torch.Tensor):
            if palette.max() > 1:
                palette = palette / 255.0
            palette = torch.clip(palette, min=0, max=1)
            colors = palette.view(num_person,
                                  3).unsqueeze(1).repeat(1, num_verts, 1)

        elif isinstance(palette, list):
            colors = []
            for person_idx in range(num_person):

                if palette[person_idx] == 'random':
                    color_person = get_different_colors(
                        num_person, int_dtype=False)[person_idx]
                    color_person = torch.FloatTensor(color_person)
                    color_person = torch.clip(
                        color_person * 1.5, min=0.6, max=1)
                    color_person = color_person.view(1, 1, 3).repeat(
                        1, num_verts, 1)
                elif palette[person_idx] == 'segmentation':
                    verts_labels = torch.zeros(num_verts)
                    color_person = torch.ones(1, num_verts, 3)
                    color_part = get_different_colors(
                        len(body_segger), int_dtype=False)
                    for part_idx, k in enumerate(body_segger.keys()):
                        index = body_segger[k]
                        verts_labels[index] = part_idx
                        color_person[:, index] = torch.FloatTensor(
                            color_part[part_idx])
                elif palette[person_idx] in Color.color_names:
                    color_person = torch.FloatTensor(
                        Color(palette[person_idx]).rgb).view(1, 1, 3).repeat(
                            1, num_verts, 1)
                else:
                    raise ValueError('Wrong palette string. '
                                     'Please choose in the pre-defined range.')
                colors.append(color_person)
            colors = torch.cat(colors, 0)
            assert colors.shape == (num_person, num_verts, 3)
            # the color passed to renderer will be (num_person, num_verts, 3)
        else:
            raise ValueError(
                'Palette should be tensor, array or list of strs.')
    return colors


def render_smpl(
    # smpl parameters
    poses: Optional[Union[torch.Tensor, np.ndarray, dict]] = None,
    betas: Optional[Union[torch.Tensor, np.ndarray]] = None,
    transl: Optional[Union[torch.Tensor, np.ndarray]] = None,
    verts: Optional[Union[torch.Tensor, np.ndarray]] = None,
    model_type: Literal['smpl', 'smplx'] = 'smpl',
    body_model: Optional[nn.Module] = None,
    body_model_config: Optional[dict] = None,
    # camera parameters
    R: Optional[Union[torch.Tensor, np.ndarray]] = None,
    T: Optional[Union[torch.Tensor, np.ndarray]] = None,
    K: Optional[Union[torch.Tensor, np.ndarray]] = None,
    orig_cam: Optional[Union[torch.Tensor, np.ndarray]] = None,
    Ks: Optional[Union[torch.Tensor, np.ndarray]] = None,
    in_ndc: bool = True,
    convention: str = 'pytorch3d',
    projection: Literal['weakperspective', 'perspective', 'fovperspective',
                        'orthographics', 'fovorthographics'] = 'perspective',
    orbit_speed: Union[float, Tuple[float, float]] = 0.0,
    # render choice parameters
    render_choice: Literal['lq', 'mq', 'hq', 'silhouette', 'depth', 'normal',
                           'pointcloud', 'part_silhouette'] = 'hq',
    palette: Union[List[str], str, np.ndarray] = 'white',
    resolution: Optional[Union[List[int], Tuple[int, int]]] = None,
    start: int = 0,
    end: Optional[int] = None,
    alpha: float = 1.0,
    no_grad: bool = True,
    batch_size: int = 10,
    device: Union[torch.device, str] = 'cuda',
    # file io parameters
    return_tensor: bool = False,
    output_path: str = None,
    origin_frames: Optional[str] = None,
    frame_list: Optional[List[str]] = None,
    image_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
    img_format: str = 'frame_%06d.jpg',
    overwrite: bool = False,
    mesh_file_path: Optional[str] = None,
    read_frames_batch: bool = False,
    # visualize keypoints
    plot_kps: bool = False,
    kp3d: Optional[Union[np.ndarray, torch.Tensor]] = None,
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

            Defaults to None.
        body_model (nn.Module, optional): body_model created from smplx.create.
            Higher priority than `body_model_config`. Should not both be None.

            Defaults to None.
        body_model_config (dict, optional): body_model_config for build_model.
            Lower priority than `body_model`. Should not both be None.
            Defaults to None.

        # camera parameters:

        K (Optional[Union[torch.Tensor, np.ndarray]], optional):
            shape should be (frame, 4, 4) or (frame, 3, 3), frame could be 1.
            if (4, 4) or (3, 3), dim 0 will be added automatically.
            Will be default `FovPerspectiveCameras` intrinsic if None.
            Lower priority than `orig_cam`.
        R (Optional[Union[torch.Tensor, np.ndarray]], optional):
            shape should be (frame, 3, 3), If f equals 1, camera will have
            identical rotation.
            If `K` and `orig_cam` is None, will be generated by `look_at_view`.
            If have `K` or `orig_cam` and `R` is None, will be generated by
            `convert_cameras`.

            Defaults to None.
        T (Optional[Union[torch.Tensor, np.ndarray]], optional):
            shape should be (frame, 3). If f equals 1, camera will have
            identical translation.
            If `K` and `orig_cam` is None, will be generated by `look_at_view`.
            If have `K` or `orig_cam` and `T` is None, will be generated by
            `convert_cameras`.

            Defaults to None.
        orig_cam (Optional[Union[torch.Tensor, np.ndarray]], optional):
            shape should be (frame, 4) or (frame, num_person, 4). If f equals
            1, will be repeated to num_frames. num_person should be 1 if single
            person. Usually for HMR, VIBE predicted cameras.
            Higher priority than `K` & `R` & `T`.

            Defaults to None.
        Ks (Optional[Union[torch.Tensor, np.ndarray]], optional):
            shape should be (frame, 4, 4).
            This is for HMR or SPIN multi-person demo.
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
            choose in ['lq', 'mq', 'hq', 'silhouette', 'depth', 'normal',
            'pointcloud', 'part_silhouette'] .

            `lq`, `mq`, `hq` would output (frame, h, w, 4) tensor.

            `lq` means low quality, `mq` means medium quality,
            h`q means high quality.

            `silhouette` would output (frame, h, w) binary tensor.

            `part_silhouette` would output (frame, h, w, n_class) tensor.

            n_class is the body segmentation classes.

            `depth` will output a depth map of (frame, h, w, 1) tensor
            and 'normal' will output a normal map of (frame, h, w, 1).

            `pointcloud` will output a (frame, h, w, 4) tensor.

            Defaults to 'mq'.
        palette (Union[List[str], str, np.ndarray], optional):
            color theme str or list of color str or `array`.

            1). If use str to represent the color,
            should choose in ['segmentation', 'random'] or color from
            Colormap https://en.wikipedia.org/wiki/X11_color_names.
            If choose 'segmentation', will get a color for each part.

            2). If you have multi-person, better give a list of str or all
            will be in the same color.

            3). If you want to define your specific color, use an `array`
            of shape (3,) for single person and (N, 3) for multiple persons.

            If (3,) for multiple persons, all will be in the same color.

            Your `array` should be in range [0, 255] for 8 bit color.

            Defaults to 'white'.
        resolution (Union[Iterable[int], int], optional):
            1). If iterable, should be (height, width) of output images.

            2). If int, would be taken as (resolution, resolution).

            Defaults to (1024, 1024).

            This will influence the overlay results when render with
            backgrounds. The output video will be rendered following the
            size of background images and finally resized to resolution.
        start (int, optional): start frame index. Defaults to 0.

        end (int, optional): end frame index. Exclusive.
                Could be positive int or negative int or None.
                None represents include all the frames.

            Defaults to None.
        alpha (float, optional): Transparency of the mesh.
            Range in [0.0, 1.0]

            Defaults to 1.0.
        no_grad (bool, optional): Set to True if do not need differentiable
            render.

            Defaults to False.
        batch_size (int, optional):  Batch size for render.
            Related to your gpu memory.

            Defaults to 10.
        # file io parameters:

        return_tensor (bool, optional): Whether return the result tensors.

            Defaults to False, will return None.
        output_path (str, optional): output video or gif or image folder.

            Defaults to None, pass export procedure.

        # background frames, priority: image_array > frame_list > origin_frames

        origin_frames (Optional[str], optional): origin background frame path,
            could be `.mp4`, `.gif`(will be sliced into a folder) or an image
            folder.

            Defaults to None.
        frame_list (Optional[List[str]], optional): list of origin background
            frame paths, element in list each should be a image path like
            `*.jpg` or `*.png`.
            Use this when your file names is hard to sort or you only want to
            render a small number frames.

            Defaults to None.
        image_array: (Optional[Union[np.ndarray, torch.Tensor]], optional):
            origin background frame `tensor` or `array`, use this when you
            want your frames in memory as array or tensor.
        overwrite (bool, optional): whether overwriting the existing files.

            Defaults to False.
        mesh_file_path (bool, optional): the directory path to store the `.ply`
            or '.ply' files. Will be named like 'frame_idx_person_idx.ply'.

            Defaults to None.
        read_frames_batch (bool, optional): Whether read frames by batch.
            Set it as True if your video is large in size.

            Defaults to False.

        # visualize keypoints
        plot_kps (bool, optional): whether plot keypoints on the output video.

            Defaults to False.
        kp3d (Optional[Union[np.ndarray, torch.Tensor]], optional):
            the keypoints of any convention, should pass `mask` if have any
            none-sense points. Shape should be (frame, )

            Defaults to None.
        mask (Optional[Union[np.ndarray, List[int]]], optional):
            Mask of keypoints existence.

            Defaults to None.
        vis_kp_index (bool, optional):
            Whether plot keypoint index number on human mesh.

            Defaults to False.
    Returns:
        Union[None, torch.Tensor]: return the rendered image tensors or None.
    """
    # initialize the device
    device = torch.device(device) if isinstance(device, str) else device

    RENDER_CONFIGS = mmcv.Config.fromfile(
        os.path.join(
            Path(mmhuman3d.__file__).parents[1],
            'configs/render/smpl.py'))['RENDER_CONFIGS']

    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    elif isinstance(resolution, list):
        resolution = tuple(resolution)

    verts, poses, betas, transl = _prepare_input_pose(verts, poses, betas,
                                                      transl)

    body_model = _prepare_body_model(model_type, body_model, body_model_config)
    model_type = body_model.name().replace('-', '').lower()
    assert model_type in ['smpl', 'smplx']

    vertices, faces, joints, num_frames, num_person = _prepare_mesh(
        poses, betas, transl, verts, start, end, body_model)
    end = num_frames if end is None else end
    vertices = vertices.view(num_frames, num_person, -1, 3)
    num_verts = vertices.shape[-2]

    if render_choice == 'pointcloud':
        plot_kps = True
    else:
        plot_kps = False

    if not plot_kps:
        joints = None
        if kp3d is not None:
            warnings.warn('`plot_kps` is False, `kp3d` will be set as None.')
            kp3d = None

    image_array, remove_folder, frames_folder = _prepare_background(
        image_array, frame_list, origin_frames, output_path, start, end,
        img_format, overwrite, num_frames, read_frames_batch)

    render_resolution = None
    if image_array is not None:
        render_resolution = (image_array.shape[1], image_array.shape[2])
    elif frames_folder is not None:
        frame_path_list = glob.glob(osp.join(
            frames_folder, '*.jpg')) + glob.glob(
                osp.join(frames_folder, '*.png')) + glob.glob(
                    osp.join(frames_folder, '*.jpeg'))
        vid_info = vid_info_reader(frame_path_list[0])
        render_resolution = (int(vid_info['height']), int(vid_info['width']))
    if resolution is not None:
        if render_resolution is not None:
            if render_resolution != resolution:
                warnings.warn(
                    f'Size of background: {render_resolution} !='
                    f' resolution: {resolution}, the output video will be '
                    f'resized as {resolution}')
            final_resolution = resolution
        elif render_resolution is None:
            render_resolution = final_resolution = resolution
    elif resolution is None:
        if render_resolution is None:
            render_resolution = final_resolution = (1024, 1024)
        elif render_resolution is not None:
            final_resolution = render_resolution

    if isinstance(kp3d, np.ndarray):
        kp3d = torch.Tensor(kp3d)

    if kp3d is not None:
        if mask is not None:
            map_index = np.where(np.array(mask) != 0)[0]
            kp3d = kp3d[map_index.tolist()]
        kp3d = kp3d[start:end]
        kp3d = kp3d.view(num_frames, -1, 3)

    # prepare render_param_dict
    render_param_dict = copy.deepcopy(RENDER_CONFIGS[render_choice.lower()])
    if model_type == 'smpl':
        render_param_dict.update(num_class=24)
    elif model_type == 'smplx':
        render_param_dict.update(num_class=27)

    if render_choice not in [
            'hq', 'mq', 'lq', 'silhouette', 'part_silhouette', 'depth',
            'pointcloud', 'normal'
    ]:
        raise ValueError('Please choose the right render_choice.')

    # body part colorful visualization should use flat shader to be sharper.
    if isinstance(palette, str):
        if (palette == 'segmentation') and ('silhouette'
                                            not in render_choice.lower()):
            render_param_dict['shader_type'] = 'flat'

        palette = [palette] * num_person
    elif isinstance(palette, np.ndarray):
        palette = torch.Tensor(palette)
        palette = palette.view(-1, 3)
        if palette.shape[0] != num_person:
            _times = num_person // palette.shape[0]
            palette = palette.repeat(_times, 1)[:num_person]
            if palette.shape[0] == 1:
                print(f'Same color for all the {num_person} people')
            else:
                print('Repeat palette for multi-person.')
    else:
        raise ValueError('Wrong input palette type. '
                         'Palette should be tensor, array or list of strs')
    colors = _prepare_colors(palette, render_choice, num_person, num_verts,
                             model_type)

    # write .ply files
    if mesh_file_path is not None:
        mmcv.mkdir_or_exist(mesh_file_path)
        for frame_idx in range(num_frames):
            ply_paths = [
                f'{mesh_file_path}/frame{frame_idx}_person{person_idx}.ply'
                for person_idx in range(num_person)
            ]
            save_meshes_as_plys(
                verts=vertices[frame_idx],
                faces=faces[None].repeat(num_person, 1, 1),
                verts_rgb=colors,
                paths=ply_paths)

    # prepare camera matrixs
    if Ks is not None:
        projection = 'perspective'
        orig_cam = None
        if isinstance(Ks, np.ndarray):
            Ks = torch.Tensor(Ks)
        Ks = Ks.view(-1, num_person, 3, 3)
        Ks = Ks[start:end]
        Ks = Ks.view(-1, 3, 3)
        K = K.repeat(num_frames * num_person, 1, 1)

        Ks = K.inverse() @ Ks @ K
        vertices = vertices.view(num_frames * num_person, -1, 3)
        if T is None:
            T = torch.zeros(num_frames, num_person, 1, 3)
        elif isinstance(T, np.ndarray):
            T = torch.Tensor(T)
        T = T[start:end]
        T = T.view(num_frames * num_person, 1, 3)
        vertices = torch.einsum('blc,bvc->bvl', Ks, vertices + T)

        R = None
        T = None
        vertices = vertices.view(num_frames, num_person, -1, 3)

    if orig_cam is not None:
        if isinstance(orig_cam, np.ndarray):
            orig_cam = torch.Tensor(orig_cam)
        projection = 'weakperspective'
        r = render_resolution[1] / render_resolution[0]
        orig_cam = orig_cam[start:end]
        orig_cam = orig_cam.view(num_frames, num_person, 4)
        # if num_person > 1:
        sx, sy, tx, ty = torch.unbind(orig_cam, -1)

        vertices[..., 0] += tx.view(num_frames, num_person, 1)
        vertices[..., 1] += ty.view(num_frames, num_person, 1)
        vertices[..., 0] *= sx.view(num_frames, num_person, 1)
        vertices[..., 1] *= sy.view(num_frames, num_person, 1)
        orig_cam = torch.tensor([1.0, 1.0, 0.0,
                                 0.0]).view(1, 4).repeat(num_frames, 1)
        K, R, T = WeakPerspectiveCameras.convert_orig_cam_to_matrix(
            orig_cam=orig_cam,
            znear=torch.min(vertices[..., 2] - 1),
            aspect_ratio=r)

    # orig_cam and K are None, use look_at_view
    if K is None:
        projection = 'fovperspective'
        K, R, T = compute_orbit_cameras(
            at=(torch.mean(vertices.view(-1, 3), 0)),
            orbit_speed=orbit_speed,
            batch_size=num_frames,
            convention=convention)
        convention = 'pytorch3d'

    if isinstance(R, np.ndarray):
        R = torch.Tensor(R).view(-1, 3, 3)
    elif isinstance(R, torch.Tensor):
        R = R.view(-1, 3, 3)
    elif isinstance(R, list):
        R = torch.Tensor(R).view(-1, 3, 3)
    elif R is None:
        pass
    else:
        raise ValueError(f'Wrong type of R: {type(R)}!')

    if R is not None:
        if len(R) > num_frames:
            R = R[start:end]

    if isinstance(T, np.ndarray):
        T = torch.Tensor(T).view(-1, 3)
    elif isinstance(T, torch.Tensor):
        T = T.view(-1, 3)
    elif isinstance(T, list):
        T = torch.Tensor(T).view(-1, 3)
    elif T is None:
        pass
    else:
        raise ValueError(f'Wrong type of T: {type(T)}!')

    if T is not None:
        if len(T) > num_frames:
            T = T[start:end]

    if isinstance(K, np.ndarray):
        K = torch.Tensor(K).view(-1, K.shape[-2], K.shape[-1])
    elif isinstance(K, torch.Tensor):
        K = K.view(-1, K.shape[-2], K.shape[-1])
    elif isinstance(K, list):
        K = torch.Tensor(K)
        K = K.view(-1, K.shape[-2], K.shape[-1])
    else:
        raise ValueError(f'Wrong type of K: {type(K)}!')

    if K is not None:
        if len(K) > num_frames:
            K = K[start:end]

    assert projection in [
        'perspective', 'weakperspective', 'orthographics', 'fovorthographics',
        'fovperspective'
    ], f'Wrong camera projection: {projection}'
    if projection in ['fovperspective', 'perspective']:
        is_perspective = True
    elif projection in [
            'fovorthographics', 'weakperspective', 'orthographics'
    ]:
        is_perspective = False
    if projection in ['fovperspective', 'fovorthographics', 'weakperspective']:
        assert in_ndc

    K, R, T = convert_cameras(
        convention_dst='pytorch3d',
        K=K,
        R=R,
        T=T,
        is_perspective=is_perspective,
        convention_src=convention,
        resolution_src=render_resolution,
        in_ndc_src=in_ndc,
        in_ndc_dst=in_ndc)

    # initialize the renderer.
    renderer = SMPLRenderer(
        resolution=render_resolution,
        faces=faces,
        device=device,
        output_path=output_path,
        colors=colors,
        return_tensor=return_tensor,
        alpha=alpha,
        model_type=model_type,
        img_format=img_format,
        render_choice=render_choice,
        projection=projection,
        frames_folder=frames_folder,
        plot_kps=plot_kps,
        vis_kp_index=vis_kp_index,
        in_ndc=in_ndc,
        final_resolution=final_resolution,
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


def visualize_smpl_calibration(
    K,
    R,
    T,
    resolution,
    **kwargs,
) -> None:
    """Visualize a smpl mesh which has opencv calibration matrix defined in
    screen."""
    assert K is not None, '`K` is required.'
    assert resolution is not None, '`resolution`(h, w) is required.'
    func = partial(
        render_smpl,
        projection='perspective',
        convention='opencv',
        orig_cam=None,
        in_ndc=False)
    for k in func.keywords.keys():
        if k in kwargs:
            kwargs.pop(k)
    return func(K=K, R=R, T=T, resolution=resolution, **kwargs)


def visualize_smpl_hmr(cam_transl,
                       bbox=None,
                       kp2d=None,
                       focal_length=5000,
                       det_width=224,
                       det_height=224,
                       bbox_format='xyxy',
                       **kwargs) -> None:
    """Simplest way to visualize HMR or SPIN or Smplify pred smpl with origin
    frames and predicted cameras."""
    if kp2d is not None:
        bbox = convert_kp2d_to_bbox(kp2d, bbox_format=bbox_format)
    Ks = convert_bbox_to_intrinsic(bbox, bbox_format=bbox_format)
    K = torch.Tensor(
        get_default_hmr_intrinsic(
            focal_length=focal_length,
            det_height=det_height,
            det_width=det_width))
    func = partial(
        render_smpl,
        projection='perspective',
        convention='opencv',
        in_ndc=False,
        K=None,
        R=None,
        orig_cam=None,
    )
    if isinstance(cam_transl, np.ndarray):
        cam_transl = torch.Tensor(cam_transl)
    T = torch.cat([
        cam_transl[..., [1]], cam_transl[..., [2]], 2 * focal_length /
        (det_width * cam_transl[..., [0]] + 1e-9)
    ], -1)
    for k in func.keywords.keys():
        if k in kwargs:
            kwargs.pop(k)
    return func(Ks=Ks, K=K, T=T, **kwargs)


def visualize_smpl_vibe(orig_cam=None,
                        pred_cam=None,
                        bbox=None,
                        output_path='sample.mp4',
                        resolution=None,
                        aspect_ratio=1.0,
                        bbox_scale_factor=1.25,
                        bbox_format='xyxy',
                        **kwargs) -> None:
    """Simplest way to visualize pred smpl with origin frames and predicted
    cameras."""
    assert resolution is not None
    if pred_cam is not None and bbox is not None:
        orig_cam = torch.Tensor(
            convert_crop_cam_to_orig_img(pred_cam, bbox, resolution[1],
                                         resolution[0], aspect_ratio,
                                         bbox_scale_factor, bbox_format))
    assert orig_cam is not None, '`orig_cam` is required.'

    func = partial(
        render_smpl,
        projection='weakperspective',
        convention='opencv',
        in_ndc=True,
    )
    for k in func.keywords.keys():
        if k in kwargs:
            kwargs.pop(k)
    return func(
        orig_cam=orig_cam,
        output_path=output_path,
        resolution=resolution,
        **kwargs)


def visualize_T_pose(num_frames,
                     orbit_speed=1.0,
                     model_type='smpl',
                     **kwargs) -> None:
    """Simplest way to visualize a sequence of T pose."""
    assert num_frames > 0, '`num_frames` is required.'
    if model_type == 'smpl':
        poses = torch.zeros(num_frames, 72)
    else:
        poses = torch.zeros(num_frames, 165)

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
        origin_frames=None)
    for k in func.keywords.keys():
        if k in kwargs:
            kwargs.pop(k)
    return func(
        poses=poses, model_type=model_type, orbit_speed=orbit_speed, **kwargs)


def visualize_smpl_pose(poses=None, verts=None, **kwargs) -> None:
    """Simplest way to visualize a sequence of smpl pose.

    Cameras will focus on the center of smpl mesh. `orbit speed` is
    recommended.
    """
    assert (poses
            is not None) or (verts
                             is not None), 'Pass either `poses` or `verts`.'
    func = partial(
        render_smpl,
        convention='opencv',
        projection='fovperspective',
        K=None,
        R=None,
        T=None,
        in_ndc=True,
        origin_frames=None,
        frame_list=None,
        image_array=None)
    for k in func.keywords.keys():
        if k in kwargs:
            kwargs.pop(k)
    return func(poses=poses, **kwargs)
