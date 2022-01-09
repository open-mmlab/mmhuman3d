import os.path as osp
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from pytorch3d.renderer.lighting import DirectionalLights, PointLights
from pytorch3d.structures import Meshes
from torch.nn.functional import interpolate

from mmhuman3d.core.conventions.segmentation import body_segmentation
from mmhuman3d.utils.ffmpeg_utils import images_to_array
from mmhuman3d.utils.mesh_utils import join_batch_meshes_as_scene
from mmhuman3d.utils.path_utils import check_path_suffix
from .base_renderer import MeshBaseRenderer
from .builder import build_renderer, build_textures

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class SMPLRenderer(MeshBaseRenderer):
    """Render SMPL(X) with different render choices."""

    def __init__(self,
                 resolution: Tuple[int, int],
                 faces: Union[np.ndarray, torch.LongTensor],
                 device: Union[torch.device, str] = 'cpu',
                 output_path: Optional[str] = None,
                 colors: Optional[Union[List[str], np.ndarray,
                                        torch.Tensor]] = None,
                 return_tensor: bool = False,
                 alpha: float = 1.0,
                 model_type='smpl',
                 out_img_format: str = '%06d.png',
                 render_choice='mq',
                 projection: Literal['weakperspective', 'fovperspective',
                                     'orthographics', 'perspective',
                                     'fovorthographics'] = 'weakperspective',
                 frames_folder: Optional[str] = None,
                 plot_kps: bool = False,
                 vis_kp_index: bool = False,
                 in_ndc: bool = True,
                 final_resolution: Tuple[int, int] = None,
                 **kwargs) -> None:
        super(MeshBaseRenderer, self).__init__()

        self.device = device
        self.projection = projection
        self.resolution = resolution
        self.model_type = model_type
        self.in_ndc = in_ndc
        self.render_choice = render_choice
        self.output_path = output_path
        self.raw_faces = torch.LongTensor(faces.astype(
            np.int32)) if isinstance(faces, np.ndarray) else faces
        self.colors = torch.Tensor(colors) if isinstance(
            colors, np.ndarray) else colors
        self.frames_folder = frames_folder
        self.plot_kps = plot_kps
        self.vis_kp_index = vis_kp_index
        self.out_img_format = out_img_format
        self.final_resolution = final_resolution
        self.segmentation = body_segmentation(self.model_type)
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

        renderer_type = kwargs.pop('renderer_type', 'base')
        self.texture_type = kwargs.pop('texture_type', 'vertex')

        self.image_renderer = build_renderer(
            dict(
                type=renderer_type,
                device=device,
                resolution=resolution,
                projection=projection,
                in_ndc=in_ndc,
                return_type=['tensor', 'rgba'] if return_tensor else ['rgba'],
                **kwargs))

        if plot_kps:
            self.alpha = max(min(0.8, alpha), 0.1)
            self.joints_renderer = build_renderer(
                dict(
                    type='pointcloud',
                    resolution=resolution,
                    device=device,
                    return_type=['rgba'],
                    projection=projection,
                    in_ndc=in_ndc,
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

    def forward(
        self,
        vertices: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        joints: Optional[torch.Tensor] = None,
        joints_gt: Optional[torch.Tensor] = None,
        indexes: Optional[Iterable[int]] = None,
    ) -> Union[None, torch.Tensor]:
        """Forward render procedure.

        Args:
            vertices (torch.Tensor): shape should be (frame, num_V, 3) or
                (frame, num_people, num_V, 3). Num people Would influence
                the visualization.
            K (Optional[torch.Tensor], optional):
                shape should be (f * 4 * 4)/perspective/orthographics or
                (f * P * 4)/weakperspective, f could be 1.
                P is person number, should be 1 if single person. Usually for
                HMR, VIBE predicted cameras.
                Defaults to None.
            R (Optional[torch.Tensor], optional):
                shape should be (f * 3 * 3).
                Will be look_at_view if None.
                Defaults to None.
            T (Optional[torch.Tensor], optional):
                shape should be (f * 3).
                Will be look_at_view if None.
                Defaults to None.
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
                be (frame, h, w, n_class) or (frame, num_people, h, w, n_class
                ). `n_class` is the number of part segments defined by smpl of
                smplx.
        """
        num_frames, num_person, _, _ = vertices.shape
        faces = self.raw_faces[None].repeat(num_frames, 1, 1)
        if self.frames_folder is not None and images is None:
            images = images_to_array(
                self.frames_folder,
                resolution=self.resolution,
                img_format=self.img_format,
                start=indexes[0],
                end=indexes[-1],
                disable_log=True).astype(np.float64)
            images = torch.Tensor(images).to(self.device)

        if images is not None:
            images = images.to(self.device)

        mesh_list = []
        for person_idx in range(num_person):
            color = self.colors[person_idx]
            verts_rgb = color[None].repeat(num_frames, 1, 1)
            mesh = Meshes(
                verts=vertices[:, person_idx].to(self.device),
                faces=faces.to(self.device),
                textures=build_textures(
                    dict(
                        type=self.texture_type,
                        verts_features=verts_rgb.to(self.device))))
            mesh_list.append(mesh)
        meshes = join_batch_meshes_as_scene(mesh_list)

        cameras = self.init_cameras(K, R, T)

        lights = getattr(self.image_renderer, 'lights', None)
        if isinstance(lights, DirectionalLights):
            lights = lights.clone()
            lights.direction = -cameras.get_camera_plane_normals()
        elif isinstance(lights, PointLights):
            lights = lights.clone()
            lights.location = -cameras.get_camera_plane_normals(
            ) - cameras.get_camera_center()

        elif lights is None:
            assert self.image_renderer.shader_type in [
                'silhouette', 'nolight', None
            ]
            lights = None
        else:
            raise TypeError(
                f'Wrong light type: {type(self.image_renderer.lights)}.')

        render_results = self.image_renderer(
            meshes=meshes, K=K, R=R, T=T, lights=lights, indexes=indexes)
        rendered_images = render_results['rgba']

        rgbs = rendered_images[..., :3]
        valid_masks = (rendered_images[..., 3:] > 0) * 1.0
        images = images / 255. if images is not None else None
        bgrs = self.rgb2bgr(rgbs)

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
                    K=K,
                    R=R,
                    T=T)['rgba']

                pointcloud_rgb = pointcloud_images[..., :3]
                pointcloud_bgr = self.rgb2bgr(pointcloud_rgb)
                pointcloud_mask = (pointcloud_images[..., 3:] > 0) * 1.0
                output_images = output_images * (
                    1 - pointcloud_mask) + pointcloud_mask * pointcloud_bgr

            output_images = self.image_tensor2numpy(output_images)

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
            rendered_map = render_results['tensor']

            if self.final_resolution != self.resolution:
                rendered_map = interpolate(
                    rendered_map, size=self.final_resolution, mode='bilinear')
            return rendered_map
        else:
            return None
