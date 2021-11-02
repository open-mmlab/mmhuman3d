import os.path as osp
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from pytorch3d.io import save_obj
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer.lighting import DirectionalLights, PointLights
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Transform3d, Translate

from mmhuman3d.utils.ffmpeg_utils import images_to_array
from mmhuman3d.utils.keypoint_utils import get_different_colors
from mmhuman3d.utils.mesh_utils import (
    join_batch_meshes_as_scene,
    mesh_to_pointcloud_vc,
)
from .base_renderer import MeshBaseRenderer
from .depth_renderer import DepthRenderer
from .pointcloud_renderer import PointCloudRenderer
from .render_factory import PALETTE, SMPL_SEGMENTATION
from .textures import TexturesClosest

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class SMPLRenderer(MeshBaseRenderer):

    def __init__(self,
                 resolution: Tuple[int, int],
                 faces: Union[np.ndarray, torch.LongTensor],
                 device: Union[torch.device, str] = 'cpu',
                 obj_path: Optional[str] = None,
                 output_path: Optional[str] = None,
                 palette: Optional[Union[List[str], np.ndarray]] = None,
                 return_tensor: bool = False,
                 alpha: float = 1.0,
                 model_type='smpl',
                 img_format: str = '%06d.png',
                 out_img_format: str = '%06d.png',
                 render_choice='mq',
                 projection: Literal['weakperspective', 'fovperspective',
                                     'orthographics', 'perspective',
                                     'fovorthographics'] = 'weakperspective',
                 frames_folder: Optional[str] = None,
                 plot_kps: bool = False,
                 vis_kp_index: bool = False,
                 **kwargs) -> None:
        if plot_kps:
            self.alpha = max(min(0.8, alpha), 0.1)
        else:
            self.alpha = max(min(1.0, alpha), 0.1)
        self.model_type = model_type
        self.render_choice = render_choice
        self.raw_faces = torch.LongTensor(faces.astype(
            np.int32)) if isinstance(faces, np.ndarray) else faces
        self.palette = palette
        self.frames_folder = frames_folder
        self.plot_kps = plot_kps
        self.vis_kp_index = vis_kp_index
        self.out_img_format = out_img_format
        super().__init__(
            resolution,
            device=device,
            obj_path=obj_path,
            output_path=output_path,
            return_tensor=return_tensor,
            alpha=alpha,
            img_format=img_format,
            projection=projection,
            **kwargs)
        if plot_kps:
            self.joints_renderer = PointCloudRenderer(
                resolution=resolution,
                device=device,
                return_tensor=True,
                projection=projection,
                radius=0.008)
        if self.render_choice == 'pointcloud':
            self.pointcloud_renderer = PointCloudRenderer(
                resolution=resolution,
                device=device,
                return_tensor=True,
                projection=projection,
                radius=0.003)
        elif self.render_choice == 'depth':
            self.depth_renderer = DepthRenderer(
                device=device,
                resolution=resolution,
                projection=projection,
                return_tensor=True)
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
            obj_path (Optional[str], optional): output .obj file directory.
                if None, would export no obj files.
                Defaults to None.
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

    def set_render_params(self, **kwargs):
        super(SMPLRenderer, self).set_render_params(**kwargs)
        self.Textures = TexturesVertex
        if self.render_choice == 'part_silhouette':
            self.slice_index = {}
            for k in SMPL_SEGMENTATION[self.model_type]['keys']:
                self.slice_index[k] = SMPL_SEGMENTATION[
                    self.model_type]['func'](
                        k)
            self.Textures = TexturesClosest

    def forward(
        self,
        vertices: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        joints: Optional[torch.Tensor] = None,
        joints_gt: Optional[torch.Tensor] = None,
        indexs: Optional[Iterable[int]] = None,
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
            indexs (Optional[Iterable[int]], optional):
                indexs for writing images.
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
        num_frame, num_person, num_verts, _ = vertices.shape
        faces = self.raw_faces[None].repeat(num_frame, 1, 1)
        if self.frames_folder is not None and images is None:
            images = images_to_array(
                self.frames_folder,
                resolution=self.resolution,
                img_format=self.img_format,
                start=indexs[0],
                end=indexs[-1],
                disable_log=True).astype(np.float64)
            images = torch.Tensor(images).to(self.device)

        if images is not None:
            images = images.to(self.device)

        mesh_list = []
        for person_idx in range(num_person):
            palette = self.palette[person_idx]

            if self.render_choice == 'silhouette':
                verts_rgb = torch.ones(num_frame, num_verts, 1)
            elif self.render_choice == 'part_silhouette':
                verts_rgb = torch.zeros(num_frame, num_verts, 1)
                for i, k in enumerate(self.slice_index):
                    verts_rgb[:, self.slice_index[k]] = 0.01 * (i + 1)
            else:
                if isinstance(palette, np.ndarray):
                    verts_rgb = torch.tensor(palette).view(1, 1, 3).repeat(
                        num_frame, num_verts, 1)
                else:
                    if palette in PALETTE:
                        verts_rgb = PALETTE[palette][None, None].repeat(
                            num_frame, num_verts, 1)
                    elif palette == 'random':
                        color = get_different_colors(num_person)[person_idx]
                        color = torch.tensor(color).float() / 255.0
                        color = torch.clip(color * 1.5, min=0.6, max=1)
                        verts_rgb = color.view(1, 1, 3).repeat(
                            num_frame, num_verts, 1)
                    elif palette == 'segmentation':
                        verts_labels = torch.zeros(num_verts)
                        verts_rgb = torch.ones(1, num_verts, 3)
                        color = get_different_colors(
                            len(
                                list(SMPL_SEGMENTATION[self.model_type]
                                     ['keys'])))
                        for part_idx, k in enumerate(
                                SMPL_SEGMENTATION[self.model_type]['keys']):
                            index = SMPL_SEGMENTATION[self.model_type]['func'](
                                k)
                            verts_labels[index] = part_idx
                            verts_rgb[:, index] = torch.tensor(
                                color[part_idx]).float() / 255
                        verts_rgb = verts_rgb.repeat(num_frame, 1, 1)
                    else:
                        raise ValueError('Wrong palette. Use numpy or str')
            mesh = Meshes(
                verts=vertices[:, person_idx].to(self.device),
                faces=faces.to(self.device),
                textures=self.Textures(
                    verts_features=verts_rgb.to(self.device)))
            mesh_list.append(mesh)
        meshes = join_batch_meshes_as_scene(mesh_list)

        # transform lights with camera extrinsic matrix
        transformation = Transform3d(device=self.device)
        transformation = transformation.compose(
            Rotate(R.permute(0, 2, 1), device=self.device))
        transformation = transformation.compose(
            Translate(-T, device=self.device))
        if isinstance(self.lights, DirectionalLights):
            lights = self.lights.clone()
            if self.projection == 'fovperspective':
                lights.direction[:, 2] *= -1
            lights.direction = transformation.transform_points(
                lights.direction)
        elif isinstance(self.lights, PointLights):
            lights = self.lights.clone()
            if self.projection == 'fovperspective':
                lights.location[:, 2] *= -1
            lights.location = transformation.transform_points(lights.location)
        else:
            raise TypeError(f'Wrong light type: {type(self.lights)}.')
        cameras = self.init_cameras(K, R, T)
        if self.render_choice == 'pointcloud':
            pointclouds = mesh_to_pointcloud_vc(meshes, alpha=1.0)
            rendered_images = self.pointcloud_renderer(
                pointclouds=pointclouds, K=K, R=R, T=T)
            # initial renderer
        elif self.render_choice == 'depth':
            rendered_images = self.depth_renderer(meshes, K=K, R=R, T=T)
        else:
            renderer = self.init_renderer(cameras, lights)

            # process render tensor and mask
            rendered_images = renderer(meshes)
        rgbs = rendered_images.clone()[..., :3] / rendered_images[
            ..., :3].max()
        valid_masks = (rendered_images[..., 3:] > 0) * 1.0
        if self.render_choice == 'part_silhouette':
            rendered_silhouettes = rgbs[None] * 100
            part_silhouettes = []
            for i in range(len(SMPL_SEGMENTATION[self.model_type]['keys'])):
                part_silhouettes.append(1.0 *
                                        (rendered_silhouettes == (i + 1)) *
                                        rendered_silhouettes / (i + 1))
            part_silhouettes = torch.cat(part_silhouettes, 0)
            alphas = part_silhouettes[..., 0].permute(1, 2, 3, 0)
        else:
            alphas = rendered_images[..., 3] / (rendered_images[..., 3] + 1e-9)

        # save .obj files
        if self.obj_path and (self.render_choice != 'part_silhouette'):
            for idx, real_idx in enumerate(indexs):
                save_obj(
                    osp.join(
                        self.obj_path,
                        Path(self.out_img_format % real_idx).stem + '.obj'),
                    meshes.verts_padded()[idx],
                    meshes.faces_padded()[idx])

        # write temp images for the output video
        if self.output_path is not None:
            if self.render_choice == 'silhouette':
                output_images = (alphas * 255).detach().cpu().numpy().astype(
                    np.uint8)

            elif self.render_choice == 'part_silhouette':
                colors = get_different_colors(alphas.shape[-1])
                output_images = colors * alphas[
                    ..., None].detach().cpu().numpy().astype(np.uint8)
                output_images = np.sum(output_images, -2)

            else:
                if images is not None:
                    output_images = rgbs * 255 * valid_masks * self.alpha + \
                        images * valid_masks * (
                            1 - self.alpha) + (1 - valid_masks) * images

                else:
                    output_images = rgbs * 255 * valid_masks * self.alpha + \
                        + 255 - valid_masks * self.alpha * 255

                if self.plot_kps:
                    joints = joints.to(self.device)
                    joints_2d = cameras.transform_points_screen(
                        joints, image_size=self.resolution)[..., :2]
                    if joints_gt is None:
                        joints_padded = joints
                        num_joints = joints_padded.shape[1]
                        joints_rgb_padded = torch.ones(
                            num_frame, num_joints, 4) * (
                                torch.tensor([0.0, 1.0, 0.0, 1.0]).view(
                                    1, 1, 4))
                    else:
                        joints_gt = joints_gt.to(self.device)
                        joints_padded = torch.cat([joints, joints_gt], dim=1)
                        num_joints = joints.shape[1]
                        num_joints_gt = joints_gt.shape[1]
                        joints_rgb = torch.ones(num_frame, num_joints, 4) * (
                            torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 1, 4))
                        joints_rgb_gt = torch.ones(
                            num_frame, num_joints_gt, 4) * (
                                torch.tensor([0.0, 0.0, 1.0, 1.0]).view(
                                    1, 1, 4))
                        joints_rgb_padded = torch.cat(
                            [joints_rgb, joints_rgb_gt], dim=1)

                    pointcloud_images = self.joints_renderer(
                        vertices=joints_padded,
                        verts_rgba=joints_rgb_padded.to(self.device),
                        K=K,
                        R=R,
                        T=T)

                    pointcloud_rgb = pointcloud_images[..., :3]
                    pointcloud_mask = (pointcloud_images[..., 3:] > 0) * 1.0
                    output_images = output_images * (
                        1 - pointcloud_mask
                    ) + pointcloud_mask * pointcloud_rgb * 255

                output_images = output_images.detach().cpu().numpy().astype(
                    np.uint8)
            for frame_idx, real_idx in enumerate(indexs):
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
                                    0.25 * self.resolution[1] / 500, [0, 0, 0],
                                    int(1 * self.resolution[1] / 1000))
                cv2.imwrite(
                    osp.join(folder, self.out_img_format % real_idx), im)

        # return
        if self.return_tensor:
            if 'silhouette' in self.render_choice:
                return alphas
            else:
                return rendered_images
        else:
            return None
