import os.path as osp
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from pytorch3d.structures import Meshes
from torch.nn.functional import interpolate

from mmhuman3d.core.cameras import MMCamerasBase
from mmhuman3d.utils.ffmpeg_utils import images_to_array
from mmhuman3d.utils.path_utils import check_path_suffix
from .base_renderer import BaseRenderer
from .builder import build_renderer
from .lights import DirectionalLights, PointLights
from .utils import align_input_to_padded, normalize, rgb2bgr, tensor2array


class SMPLRenderer(BaseRenderer):
    """Render SMPL(X) with different render choices."""

    def __init__(self,
                 resolution: Tuple[int, int] = None,
                 device: Union[torch.device, str] = 'cpu',
                 output_path: Optional[str] = None,
                 return_tensor: bool = False,
                 alpha: float = 1.0,
                 out_img_format: str = '%06d.png',
                 read_img_format: str = None,
                 render_choice='mq',
                 frames_folder: Optional[str] = None,
                 plot_kps: bool = False,
                 vis_kp_index: bool = False,
                 final_resolution: Tuple[int, int] = None,
                 **kwargs) -> None:
        super(BaseRenderer, self).__init__()

        self.device = device
        self.resolution = resolution
        self.render_choice = render_choice
        self.output_path = output_path
        self.frames_folder = frames_folder
        self.plot_kps = plot_kps
        self.vis_kp_index = vis_kp_index
        self.read_img_format = read_img_format
        self.out_img_format = out_img_format
        self.final_resolution = final_resolution
        self.return_tensor = return_tensor
        if output_path is not None:
            if check_path_suffix(output_path, ['.mp4', '.gif']):
                self.temp_path = osp.join(
                    Path(output_path).parent,
                    Path(output_path).name + '_output_temp')
                mmcv.mkdir_or_exist(self.temp_path)
                print('make dir', self.temp_path)
            else:
                self.temp_path = output_path

        self.image_renderer = build_renderer(
            dict(device=device, resolution=resolution, **kwargs))

        if plot_kps:
            self.alpha = max(min(0.8, alpha), 0.1)
            self.joints_renderer = build_renderer(
                dict(
                    type='pointcloud',
                    resolution=resolution,
                    device=device,
                    radius=0.008))
        else:
            self.alpha = max(min(1.0, alpha), 0.1)
        """
        Render Mesh for SMPL and SMPL-X. For function render_smpl.
        2 modes: mesh render with different quality and palette,
        or silhouette render.

        Args:
            resolution (Iterable[int]): (height, width of render images)
            faces (Union[np.ndarray, torch.LongTensor]): face of mesh to
                be rendered.
            device (torch.device, optional): cuda or cpu device.
                Defaults to torch.device('cpu').
            output_path (Optional[str], optional): render output path.
                could be .mp4 or .gif or a folder.
                Else: 1). If `render_choice` in ['lq', 'mq', 'hq'], the output
                video will be a smpl mesh video which each person in a single
                color.
                2). If `render_choice` is `silhouette`, the output video will
                be a black-white smpl silhouette video.
                3). If `render_choice` is  `part_silhouette`, the output video
                will be a smpl mesh video which each body-part in a single
                color.
                If None, no video will be wrote.
                Defaults to None.
            palette (Optional[List[str]], optional):
                List of palette string. Defaults to ['blue'].
            return_tensor (bool, optional): Whether return tensors.
                return None if set to False.
                Defaults to False.
            alpha (float, optional): transparency value, from 0.0 to 1.0.
                Defaults to 1.0.

        Returns:
            None
        """

    def to(self, device):
        return super(BaseRenderer, self).to(device)

    def forward(
        self,
        meshes: Meshes,
        cameras: Optional[MMCamerasBase] = None,
        images: Optional[torch.Tensor] = None,
        joints: Optional[torch.Tensor] = None,
        joints_gt: Optional[torch.Tensor] = None,
        indexes: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> Union[None, torch.Tensor]:
        """Forward render procedure.

        Args:
            vertices (torch.Tensor): shape should be (frame, num_V, 3) or
                (frame, num_people, num_V, 3). Num people Would influence
                the visualization.
            images (Optional[torch.Tensor], optional): Tensor of background
                images. If None, no background.
                Defaults to None.
            joints (Optional[torch.Tensor], optional):
                joints produced from smpl model.
                Defaults to None.
            joints_gt (Optional[torch.Tensor], optional):
                ground-truth points passed.
                Defaults to None.
            indexes (Optional[Iterable[int]], optional):
                indexes for writing images.
                Defaults to None.

        Returns:
            Union[None, torch.Tensor]:
                return None if not return_tensor.
                Else: 1). If render images, the output tensor shape would be
                (frame, h, w, 4) or (frame, num_people, h, w, 4), depends on
                number of people.
                2). If render silhouette, the output tensor shape will be
                (frame, h, w) or (frame, num_people, h, w).
                3). If render part silhouette, the output tensor shape should
                be (frame, h, w, 1) or (frame, num_people, h, w, 1
                ).
        """
        num_frames = len(meshes)
        if self.frames_folder is not None and images is None:

            images = images_to_array(
                self.frames_folder,
                resolution=self.resolution,
                img_format=self.read_img_format,
                start=indexes[0],
                end=indexes[-1] + 1,
                disable_log=True).astype(np.float64)
            images = torch.Tensor(images).to(self.device)
            images = align_input_to_padded(
                images,
                ndim=4,
                batch_size=num_frames,
                padding_mode='ones',
            )
        if images is not None:
            images = images.to(self.device)

        lights = getattr(self.image_renderer, 'lights', None)
        if isinstance(lights, DirectionalLights):
            lights = lights.clone()
            lights.direction = -cameras.get_camera_plane_normals()
        elif isinstance(lights, PointLights):
            lights = lights.clone()
            lights.location = -cameras.get_camera_plane_normals(
            ) - cameras.get_camera_center()

        rendered_tensor = self.image_renderer(
            meshes=meshes, cameras=cameras, lights=lights, indexes=indexes)

        rendered_images = self.image_renderer.tensor2rgba(rendered_tensor)

        rgbs = rendered_images[..., :3]
        valid_masks = rendered_images[..., 3:]
        images = normalize(
            images,
            origin_value_range=[0, 255],
            out_value_range=[0, 1],
            dtype=torch.float32) if images is not None else None

        bgrs = rgb2bgr(rgbs)

        # write temp images for the output video
        if self.output_path is not None:

            if images is not None:
                output_images = bgrs * valid_masks * self.alpha + \
                    images * valid_masks * (
                        1 - self.alpha) + (1 - valid_masks) * images

            else:
                output_images = bgrs

            if self.plot_kps:

                joints = joints.to(self.device)
                joints_2d = cameras.transform_points_screen(
                    joints, image_size=self.resolution)[..., :2]
                if joints_gt is None:
                    joints_padded = joints
                    num_joints = joints_padded.shape[1]
                    joints_rgb_padded = torch.ones(
                        num_frames, num_joints, 4) * (
                            torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 1, 4))
                else:
                    joints_gt = joints_gt.to(self.device)
                    joints_padded = torch.cat([joints, joints_gt], dim=1)
                    num_joints = joints.shape[1]
                    num_joints_gt = joints_gt.shape[1]
                    joints_rgb = torch.ones(num_frames, num_joints, 4) * (
                        torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 1, 4))
                    joints_rgb_gt = torch.ones(
                        num_frames, num_joints_gt, 4) * (
                            torch.tensor([1.0, 0.0, 0.0, 1.0]).view(1, 1, 4))
                    joints_rgb_padded = torch.cat([joints_rgb, joints_rgb_gt],
                                                  dim=1)

                pointcloud_images = self.joints_renderer(
                    vertices=joints_padded,
                    verts_rgba=joints_rgb_padded.to(self.device),
                    cameras=cameras)

                pointcloud_rgb = pointcloud_images[..., :3]
                pointcloud_bgr = rgb2bgr(pointcloud_rgb)
                pointcloud_mask = (pointcloud_images[..., 3:] > 0) * 1.0
                output_images = output_images * (
                    1 - pointcloud_mask) + pointcloud_mask * pointcloud_bgr

            output_images = tensor2array(output_images)

            for frame_idx, real_idx in enumerate(indexes):
                folder = self.temp_path if self.temp_path is not None else\
                    self.output_path
                im = output_images[frame_idx]
                if self.plot_kps and self.vis_kp_index:
                    point_xy = joints_2d[frame_idx]
                    for j_idx in range(point_xy.shape[-2]):
                        x = point_xy[j_idx, 0]
                        y = point_xy[j_idx, 1]
                        cv2.putText(im, str(j_idx), (int(x), int(y)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.25 * self.final_resolution[1] / 500,
                                    [0, 0, 0],
                                    int(1 * self.final_resolution[1] / 1000))
                if self.final_resolution != self.resolution:
                    im = cv2.resize(im, self.final_resolution, cv2.INTER_CUBIC)
                cv2.imwrite(
                    osp.join(folder, self.out_img_format % real_idx), im)

        # return
        if self.return_tensor:

            if images is not None:
                rendered_map = torch.tensor(output_images)
            else:
                rendered_map = rendered_tensor

            if self.final_resolution != self.resolution:
                rendered_map = interpolate(
                    rendered_map, size=self.final_resolution, mode='bilinear')
            return rendered_map
        else:
            return None
