import copy
import os
import os.path as osp
from functools import partial
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from configs.render.smpl import RENDER_CONFIGS
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmhuman3d.core.utils.smpl_utils import (
    get_body_model,
    get_mesh_info,
    smpl_dict2tensor,
    smpl_tensor2dict,
    smplx_dict2tensor,
    smplx_tensor2dict,
)
from mmhuman3d.core.visualization.renderer.torch3d_renderer import (
    RenderDataset,
    SMPLRenderer,
)
from mmhuman3d.utils.ffmpeg_utils import images_to_array, video_to_images


def render_smpl(
    poses: Union[torch.Tensor, np.ndarray, dict],
    betas: Optional[Union[torch.Tensor, np.ndarray]] = None,
    translations: Optional[Union[torch.Tensor, np.ndarray]] = None,
    cameras: Optional[Union[torch.Tensor, np.ndarray]] = None,
    body_model_dir: str = '',
    render_choice: str = 'hq',
    return_tensor: bool = False,
    output_path: str = None,
    model_type: str = 'smpl',
    palette: Union[List[str], str] = 'white',
    resolution: Union[Iterable[int]] = (1024, 1024),
    start: int = 0,
    end: int = -1,
    origin_frames: Optional[str] = None,
    gender: str = 'neutral',
    batch_size: int = 20,
    DDP: bool = False,
    force: bool = False,
    obj_path: Optional[str] = None,
    alpha: float = 1.0,
    no_grad: bool = False,
) -> Union[None, torch.Tensor]:
    """Render SMPL or SMPL-X mesh or silhouette into differentiable tensors,
    and export video or images.

    Args:
        poses (Union[torch.Tensor, np.ndarray]):
                should be a tensor of (frame * P * 72) for smpl.
                If single-person, P equals 1. Multi-person render should
                be fed together with multi-person weakperspective cameras.
                Required.
        betas (Optional[Union[torch.Tensor, np.ndarray]], optional):
                should be a tensor of (frame * P * 10)
                for SMPL. If poses are single person, betas would be sliced.
                If poses are multi-person, betas would be set to same person
                number. If None, would be set as torch.zeros.
                Defaults to None.
        translations (Optional[Union[torch.Tensor, np.ndarray]], optional):
                Translations. Defaults to None.
        cameras (Optional[Union[torch.Tensor, np.ndarray]], optional):
                shape should be (3 * 4) ([R | T] matrix, perpective)
                or (3 * 3) ([R] matrix, T=0, perpective),
                or (P * 4) ([scale_x, scale_y, t_x, t_y], weakperspective).
                P is person number, should be 1 if single person.
                Do not support moving cameras(not useful) yet.
                Defaults to None (will be look_at_view).
        body_model_dir (str, optional): Directory of npz or pkl path.
                Defaults to '/mnt/lustre/share/sugar/SMPLmodels/'.
        render_choice (str, optional):
                choose in ['lq', 'mq', 'hq', 'silhouette', 'part_silhouette'].
                lq, mq, hq would output (frame * h * w * 3) tensor or rgb
                frames. lq means low quality, mq means medium quality,
                hq means high quality.
                silhouette would output (frame * h * w) tensor or binary
                frames.
                part_silhouette would output (frame * h * w * n_class) tensor.
                n_class is the body segmentation classes.
                Defaults to 'mq'.
        return_tensor (bool, optional): Whether return the result tensors.
                Defaults to False, would return None.
        output_path (str, optional): output video or gif or image folder.
                Defaults to None, pass export procedure.
        model_type (str, optional): choose in 'smpl' or 'smplx'.
                Defaults to 'smpl'.
        palette (Union[List[str], str], optional): palette theme str or
                list of str. Should choose in ['segmentation', 'blue',
                'red', 'white', 'black', 'green', 'yellow', 'random']
                If choose 'segmentation', will get a palette for each part.
                If multi-person, better give a list or all will be in the
                same color.
                Defaults to 'blue'.
        start (int, optional): start frame index. Defaults to 0.
        end (int, optional): end frame index. Defaults to -1.
        origin_frames (Optional[str], optional): origin brackground frame path,
                could be .mp4(will be sliced into a folder) or an image folder.
                Defaults to None, will not paste background.
        gender (str, optional): chose in ['male', 'female', 'neutral'].
                Defaults to 'neutral'.
        batch_size (int, optional):  batch size. Defaults to 20.
        DDP (bool, optional): whether use distributeddataparallel.
                Defaults to False.
        force (bool, optional): whether replace the existing file.
                Defaults to False.
        obj_path (bool, optional): the directory path to store the .obj files.
                Defaults to None.
        alpha (float, optional): transparency of the mesh. Defaults to 1.0.
        no_grad (bool, optional): [description]. Defaults to False.
    Returns:
        Union[None, torch.Tensor]: return the rendered image tensors or None.
    """
    # initialize the gpu device
    gpu_count = int(torch.cuda.device_count())
    gpu_list = [i for i in range(gpu_count)]
    gpu_str = ','.join(list(map(lambda x: str(x), gpu_list)))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # check the output path
    if output_path is not None:
        if Path(output_path).is_file() and not force:
            raise FileExistsError(
                f'{output_path} exists (set force to  True to overwrite).')
        if not Path(output_path).parent.is_dir():
            raise NotADirectoryError(
                'The parent directory of output video does not exist.')
    # read the origin frames as array if any.
    image_array = None
    if origin_frames is not None:
        if Path(origin_frames).suffix in ['.mp4']:
            if not Path(origin_frames).is_file():
                raise FileNotFoundError('Input video does not exist.')
            else:
                temp_folder = osp.join(
                    Path(output_path).parent,
                    Path(output_path).name + '_input_temp')
                if not Path(temp_folder).is_dir():
                    os.mkdir(temp_folder)
                video_to_images(origin_frames, temp_folder)
                image_array = images_to_array(temp_folder)
        else:
            if not Path(origin_frames).is_dir():
                raise FileNotFoundError('Input frame folder does not exist.')
            else:
                temp_folder = None
                image_array = images_to_array(origin_frames)

    # prepare encode decode functions, and check the format of input poses.
    if model_type == 'smpl':
        NUM_JOINTS = 23
        NUM_BODY_JOINTS = 23
        NUM_DIM = 3 * (NUM_JOINTS + 1)
        enc_func = smpl_dict2tensor
        dec_func = smpl_tensor2dict
        if isinstance(poses, dict):
            if not {'body_pose', 'global_orient'}.issubset(poses):
                raise KeyError(
                    "Please make sure that your input dict has 'body_pose'"
                    "and 'global_orient'.")
        elif isinstance(poses, torch.Tensor):
            if poses.shape[-1] != 3 * (NUM_JOINTS + 1):
                raise ValueError(
                    f'Please make sure your poses is f{NUM_DIM} dims in'
                    'the last axis.')
        elif isinstance(poses, np.ndarray):
            if poses.shape[-1] != NUM_DIM:
                raise ValueError(
                    f'Please make sure your poses is f{NUM_DIM} dims in'
                    'the last axis.')
            else:
                poses = torch.FloatTensor(poses)
    elif model_type == 'smplx':
        NUM_JOINTS = 54
        NUM_BODY_JOINTS = 21
        NUM_DIM = 3 * (NUM_JOINTS + 1)
        enc_func = smplx_dict2tensor
        dec_func = smplx_tensor2dict
        if isinstance(poses, dict):
            if not {
                    'global_orient', 'body_pose', 'left_hand_pose',
                    'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose'
            }.issubset(poses):
                raise KeyError(
                    '%s' % str(poses.keys()),
                    'Please make sure that your input dict has all of'
                    "'global_orient', 'body_pose', 'left_hand_pose',"
                    "'right_hand_pose', 'jaw_pose', 'leye_pose',"
                    "'reye_pose'")
        elif isinstance(poses, torch.Tensor):
            if poses.shape[-1] != NUM_DIM:
                raise ValueError(
                    f'Please make sure your poses is f{NUM_DIM} dims in'
                    'the last axis.')
        elif isinstance(poses, np.ndarray):
            if poses.shape[-1] != NUM_DIM:
                raise ValueError(
                    f'Please make sure your poses is f{NUM_DIM} dims in'
                    'the last axis.')
            else:
                poses = torch.FloatTensor(poses)

    # slice the input poses, betas, and translations.
    if isinstance(poses, dict):
        num_frame = poses['body_pose'].shape[0]
        _, num_person, _ = poses['body_pose'].view(num_frame, -1,
                                                   NUM_BODY_JOINTS * 3).shape
        full_pose = enc_func(poses)
        end = (min(end, num_frame - 1) + num_frame) % num_frame
        full_pose = full_pose[start:end + 1]
        betas = betas[start:end + 1] if betas is not None else None
        translations = translations[start:end +
                                    1] if translations is not None else None
        pose_dict = dec_func(
            full_pose=full_pose, betas=betas, transl=translations)
        num_frame = full_pose.shape[0]
    elif isinstance(poses, torch.Tensor) or isinstance(poses, np.ndarray):
        poses = poses.view(poses.shape[0], -1, (NUM_JOINTS + 1) * 3)
        num_frame, num_person, _ = poses.shape
        end = (min(end, num_frame - 1) + num_frame) % num_frame
        full_pose = poses[start:end + 1]
        betas = betas[start:end + 1] if betas is not None else None
        translations = translations[start:end +
                                    1] if translations is not None else None
        pose_dict = dec_func(
            full_pose=full_pose, betas=betas, transl=translations)
        num_frame = full_pose.shape[0]
    if image_array is not None:
        image_array = image_array[start:end + 1]

    if not (Path(body_model_dir).is_dir() or Path(body_model_dir).is_file()):
        raise FileNotFoundError('Wrong body_model_dir.'
                                ' File or directory does not exist.')
    body_model = get_body_model(
        model_folder=body_model_dir,
        model_type=model_type,
        gender=gender,
        batch_size=num_frame * num_person)
    if betas is not None:
        assert betas.shape[0] == num_frame
        assert betas.shape[1] == num_person
    mesh_info = get_mesh_info(
        body_model=body_model,
        data_type='tensor',
        required_keys=['vertices', 'faces'],
        **pose_dict)
    vertices, faces = mesh_info['vertices'], mesh_info['faces']
    vertices = vertices.view(num_frame, num_person, -1, 3)

    # prepare render_param_dict
    render_choice = render_choice.lower()
    if render_choice not in [
            'hq', 'mq', 'lq', 'silhouette', 'part_silhouette'
    ]:
        raise ValueError('Please choose the right render_choice!')
    render_param_dict = copy.deepcopy(RENDER_CONFIGS[render_choice])
    render_param_dict['render_choice'] = render_choice
    render_param_dict['model_type'] = model_type

    # body part colorful visualization should use flat shader to be shaper.
    if isinstance(palette, str):
        if (palette == 'segmentation') and ('silhouette' not in render_choice):
            render_param_dict['shader']['shader_type'] = 'flat'
        palette = [palette] * num_person

    # TODO: complete multi-person visualization and get each person specific
    # color.
    if not len(palette) == num_person:
        raise ValueError('Please give the right number of palette.')

    # slice the input cameras and check the type and shape.
    if cameras is None:
        camera_type = 'perspective'
    else:
        if cameras.shape[-1] == 3:
            camera_type = 'perspective'
        elif cameras.shape[-1] == 4:
            camera_type = 'weakperspective'
        cameras = cameras[start:end + 1]
    render_param_dict['camera']['camera_type'] = camera_type

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
        **render_param_dict)
    renderer = renderer.to(device)

    # prepare the render data.
    render_dataset = RenderDataset(
        images=image_array, vertices=vertices, cameras=cameras)

    RenderLoader = DataLoader(
        dataset=render_dataset, batch_size=batch_size, shuffle=False)

    # TODO:complete DDP for multi-gpu rendering.
    if len(gpu_list) > 1:
        renderer = nn.parallel.DataParallel(renderer, device_ids=gpu_list)

    # start rendering. no grad if non-differentiable render.
    # return None if only need video.
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
    if return_tensor:
        results = torch.cat(results, 0)
        return results
    else:
        return None


def neural_render_silhouette(poses,
                             betas=None,
                             cameras=None,
                             translations=None,
                             body_model_dir='',
                             model_type='smpl',
                             gender='neutral',
                             resolution=(1024, 1024),
                             batch_size=20) -> torch.Tensor:
    """Neural render silhouette of SMPL(X) mesh.

    You could specify these following parameters:
            poses,  betas, translations, cameras, body_model_dir,
            model_type, resolution, start, end, gender, batch_size.
    Returns:
        torch.Tensor: Would be shape of (frame, H, W, 1)
    """
    func = partial(
        render_smpl,
        render_choice='silhouette',
        return_tensor=True,
        obj_path=None,
        origin_frames=None,
        output_path=None,
        no_grad=False,
        start=0,
        end=-1,
        palette='white',
        force=False)

    return func(
        poses=poses,
        betas=betas,
        cameras=cameras,
        translations=translations,
        body_model_dir=body_model_dir,
        model_type=model_type,
        gender=gender,
        resolution=resolution,
        batch_size=batch_size)


def neural_render_part_silhouette(**kwargs):
    # TODO
    pass


def neural_render_smpl(**kwargs):
    # TODO
    pass


def visualize_silhouette(**kwargs):
    # TODO
    pass


def visualize_part_silhouette(**kwargs):
    # TODO
    pass


def visualize_smpl_frame(**kwargs):
    # TODO
    pass


def visualize_smpl_pose(poses,
                        body_model_dir,
                        output_path,
                        model_type='smpl',
                        render_choice='mq',
                        force=False,
                        resolution=(1024, 1024)) -> NoReturn:
    """Simpliest way to visualize a sequence of smpl pose. render_choice should
    be in 'hq', 'mq', 'lq'.

    Returns:
        NoReturn
    """
    if render_choice.lower() not in ['hq', 'mq', 'lq']:
        raise ValueError("Please choose render_choice in ['hq', 'mq', 'lq'].")
    func = partial(
        render_smpl,
        betas=None,
        translations=None,
        cameras=None,
        return_tensor=False,
        no_grad=True,
        obj_path=None,
        start=0,
        end=-1,
        origin_frames=None,
        gender='neutral')
    func(
        poses=poses,
        body_model_dir=body_model_dir,
        model_type=model_type,
        render_choice=render_choice,
        output_path=output_path,
        force=force,
        resolution=resolution,
    )
