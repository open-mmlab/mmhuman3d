import math
import os.path as osp
import shutil
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Union

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors
from pytorch3d.common.types import Device
from pytorch3d.io import save_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    HardFlatShader,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftGouraudShader,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    hard_rgb_blend,
    look_at_view_transform,
)
from pytorch3d.renderer.cameras import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Transform3d
from torch.utils.data import Dataset

from mmhuman3d.core.conventions.segmentation.smpl import (
    SMPL_SEGMENTATION_DICT,
    smpl_part_segmentation,
)
from mmhuman3d.core.conventions.segmentation.smplx import (
    SMPLX_SEGMENTATION_DICT,
    smplx_part_segmentation,
)
from mmhuman3d.utils.cameras import WeakPerspectiveCameras
from mmhuman3d.utils.ffmpeg_utils import images_to_gif, images_to_video
from mmhuman3d.utils.keypoint_utils import get_different_colors
from mmhuman3d.utils.transforms import ee_to_rotmat


class NoLightShader(nn.Module):

    def __init__(self,
                 device: Device = 'cpu',
                 blend_params: Optional[BlendParams] = None,
                 **kwargs) -> None:
        super().__init__()
        self.blend_params = blend_params if blend_params is not None\
            else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get('blend_params', self.blend_params)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images


class TexturesCloset(TexturesVertex):

    def sample_textures(self, fragments, faces_packed=None) -> torch.Tensor:
        """Rewrite sample_textures to use the closet interpolation.

        This function would only be called in render forwarding.
        """
        verts_features_packed = self.verts_features_packed()
        faces_verts_features = verts_features_packed[faces_packed]
        bary_coords = fragments.bary_coords
        _, idx = torch.max(bary_coords, -1)
        mask = torch.arange(bary_coords.size(-1)).reshape(1, 1, -1).to(
            self.device) == idx.unsqueeze(-1)
        bary_coords *= 0
        bary_coords[mask] = 1
        texels = interpolate_face_attributes(fragments.pix_to_face,
                                             bary_coords, faces_verts_features)
        return texels


SHADER_FACTORY = {
    'phong': SoftPhongShader,
    'gouraud': SoftGouraudShader,
    'silhouette': SoftSilhouetteShader,
    'flat': HardFlatShader,
    'nolight': NoLightShader,
}

CAMERA_FACTORY = {
    'perspective': FoVPerspectiveCameras,
    'orthographic': FoVOrthographicCameras,
    'weakperspective': WeakPerspectiveCameras,
}

LIGHTS_FACTORY = {'directional': DirectionalLights, 'point': PointLights}

PALETTE = {
    'white': torch.FloatTensor([1, 1, 1]),
    'black': torch.FloatTensor([0, 0, 0]),
    'blue': torch.FloatTensor([1, 0, 0]),
    'green': torch.FloatTensor([0, 1, 0]),
    'red': torch.FloatTensor([0, 0, 1]),
    'yellow': torch.FloatTensor([0, 1, 1])
}

SEGMENTATION = {
    'smpl': {
        'keys': SMPL_SEGMENTATION_DICT.keys(),
        'func': smpl_part_segmentation
    },
    'smplx': {
        'keys': SMPLX_SEGMENTATION_DICT.keys(),
        'func': smplx_part_segmentation
    }
}


class MeshBaseRenderer(nn.Module):

    def __init__(self,
                 resolution: Iterable[int],
                 device: torch.device = torch.device('cpu'),
                 output_path: Optional[str] = None,
                 return_tensor: bool = False,
                 **kwargs) -> NoReturn:
        """MeshBaseRenderer for neural rendering and visualization.

        Returns:
            NoReturn
        """
        super().__init__()
        self.device = device
        self.conv = nn.Conv2d(1, 1, 1)
        self.output_path = output_path
        self.return_tensor = return_tensor
        self.resolution = resolution
        self.temp_path = None
        if output_path is not None:
            if Path(output_path).suffix in ['.gif', '.mp4']:
                self.temp_path = osp.join(
                    Path(output_path).parent,
                    Path(output_path).name + '_output_temp')
                mmcv.mkdir_or_exist(self.temp_path)
                print('make dir', self.temp_path)
        self.set_render_params(**kwargs)

    def set_render_params(self, **kwargs):
        material_params = kwargs.get('material')
        light_params = kwargs.get('light')
        shader_params = kwargs.get('shader')
        raster_params = kwargs.get('raster')
        camera_params = kwargs.get('camera')
        blend_params = kwargs.get('blend')
        assert light_params is not None
        assert shader_params is not None
        assert raster_params is not None
        assert camera_params is not None
        assert material_params is not None
        assert blend_params is not None
        self.shader = SHADER_FACTORY[shader_params.pop('shader_type', 'phong')]

        self.materials = Materials(device=self.device, **material_params)
        default_resolution = raster_params.pop('resolution', None)
        if self.resolution is None:
            self.resolution = default_resolution

        self.raster_settings = RasterizationSettings(
            image_size=self.resolution, **raster_params)
        light_type = light_params.pop('light_type', 'directional')
        lights = LIGHTS_FACTORY[light_type]
        self.axis_sign = camera_params.get('axis_sign')
        self.normalize = Transform3d()
        rotate = Rotate(
            R=ee_to_rotmat(
                torch.FloatTensor(self.axis_sign).view(1, 3) / 180 * math.pi))
        self.normalize = self.normalize.compose(rotate)

        self.lights = lights(device=self.device, **light_params)

        self.camera_type = camera_params.get('camera_type', 'weakperspective')
        self.camera_register = CAMERA_FACTORY[self.camera_type]
        self.blend_params = BlendParams(**blend_params)

    def export(self):
        if self.output_path is not None:
            folder = self.temp_path if self.temp_path is not None else\
                 self.output_path
            if Path(self.output_path).suffix == '.mp4':
                images_to_video(
                    input_folder=folder, output_path=self.output_path)
            elif Path(self.output_path).suffix == '.gif':
                images_to_gif(
                    input_folder=folder, output_path=self.output_path)

    def __del__(self):
        self.removeTempFrames()

    def removeTempFrames(self):
        if self.temp_path:
            if osp.exists(self.temp_path) and osp.isdir(self.temp_path):
                shutil.rmtree(self.temp_path)

    def forward(self,
                meshes: Meshes,
                camera_matrix: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                file_names: Iterable[str] = []):
        if camera_matrix is not None:

            camera_matrix = camera_matrix.to(self.device)

            if self.camera_type == 'weakperspective':
                camera_params = {
                    'scale_x': camera_matrix[..., 0],
                    'scale_y': camera_matrix[..., 1],
                    'trans_x': camera_matrix[..., 2],
                    'trans_y': camera_matrix[..., 3]
                }
            else:
                R = camera_matrix[..., :3].T
                T = camera_matrix[..., 3].T
                camera_params = {
                    'device': self.device,
                    'R': R,
                    'T': T,
                    'zfar': 1000.0
                }
        else:
            elev = 0
            azim = 0
            R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
            camera_params = {
                'device': self.device,
                'R': R,
                'T': T,
                'zfar': 1000.0
            }
        cameras = self.camera_register(**camera_params)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=self.raster_settings),
            shader=self.shader(
                device=self.device,
                cameras=cameras,
                lights=self.lights,
                materials=self.materials,
                blend_params=self.blend_params) if
            (self.shader is not SoftSilhouetteShader) else self.shader())

        rendered_images = renderer(meshes)
        rgbs, valid_masks = rendered_images[
            ..., :3], (rendered_images[..., 3:] > 0) * 1.0

        if self.output_path is not None:
            if images is not None:
                output_images = rgbs * 255 * valid_masks + (
                    1 - valid_masks) * images
                output_images = output_images.detach().cpu().numpy().astype(
                    np.uint8)
            else:
                output_images = (rgbs.detach().cpu().numpy() * 255).astype(
                    np.uint8)
            for index in range(output_images.shape[0]):
                folder = self.temp_path if self.temp_path is not None else\
                    self.output_path
                cv2.imwrite(
                    osp.join(folder, file_names[index]), output_images[index])

        if self.return_tensor:
            return rendered_images
        else:
            return None


class SMPLRenderer(MeshBaseRenderer):

    def __init__(self,
                 resolution: Iterable[int],
                 faces: Union[np.ndarray, torch.LongTensor],
                 device: torch.device = torch.device('cpu'),
                 obj_path: Optional[str] = None,
                 output_path: Optional[str] = None,
                 palette: Optional[List[str]] = ['white'],
                 return_tensor: bool = False,
                 alpha: float = 1.0,
                 **kwargs) -> NoReturn:
        super().__init__(
            resolution,
            device=device,
            output_path=output_path,
            return_tensor=return_tensor,
            alpha=alpha,
            **kwargs)
        self.obj_path = obj_path
        self.alpha = max(min(1.0, alpha), 0.1)
        self.raw_faces = torch.LongTensor(faces.astype(
            np.int32)) if isinstance(faces, np.ndarray) else faces
        self.palette = palette
        """Render for SMPL and SMPL-X.

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
                    Defaults to None.
            palette (Optional[List[str]], optional):
                    List of palette string. Defaults to ['blue'].
            return_tensor (bool, optional): Whether return tensors.
                    return None if set to False.
                    Defaults to False.
            alpha (float, optional): transparency value, from 0.0 to 1.0.
                    Defaults to 1.0.

        Returns:
            NoReturn
        """

    def set_render_params(self, **kwargs):
        super(SMPLRenderer, self).set_render_params(**kwargs)
        self.model_type = kwargs.get('model_type', 'smpl')
        self.render_choice = kwargs.get('render_choice', 'mq')
        if self.render_choice == 'part_silhouette':
            self.slice_index = {}
            for k in SEGMENTATION[self.model_type]['keys']:
                self.slice_index[k] = SEGMENTATION[self.model_type]['func'](k)

    def forward(self,
                vertices: torch.Tensor,
                camera_matrix: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                file_names: Iterable[str] = []) -> Union[None, torch.Tensor]:
        """Forward render procedure on GPUs.

        Args:
            vertices (torch.Tensor): shape should be (frame, num_V, 3) or
                    (frame, num_people, num_V, 3). Num people Would influence
                    the visualization.
            camera_matrix (Optional[torch.Tensor], optional):
                    external camera matrix. If None, would be generated by
                    look_at_view with default elevation and angle_zim,
                    If Tensor, shape should be be (frame, 3, 3). if frame is 1,
                    all the frames would share the same camera matrix.
                    If with background, the camera matrix should be computed
                    exactly.
                    Defaults to None.
            images (Optional[torch.Tensor], optional): Tensor of background
                    images. If None, no background.
                    Defaults to None.
            file_names (Iterable[str], optional): File formated name for
                    ffmpeg reading and writing.
                    Defaults to [].

        Returns:
            Union[None, torch.Tensor]:
                return None if not return_tensor.
                Else: 1). If render images, the output tensor shape would be
                (frame, h, w, 4) or (frame, num_people, h, w, 4), depends on
                number of people.
                2). If render silhouette, the output tensor shape would be
                (frame, h, w) or (frame, num_people, h, w).
                3). If render part silhouette, the output tensor shape should
                be (frame, h, w, n_class) or (frame, num_people, h, w, n_class
                ). n_class is the number of part segments defined by smpl of
                smplx.
        """
        num_frame, num_person, num_verts, _ = vertices.shape
        faces = self.raw_faces[None].repeat(num_frame, 1, 1)

        if images is not None:
            images = images.to(self.device)

        vertices = self.normalize.transform_points(
            vertices.view(-1, num_verts, 3)).view(num_frame, num_person,
                                                  num_verts, 3)

        Textures = TexturesVertex
        verts_pad = []
        verts_features_pad = []
        faces_pad = []

        for person_idx in range(num_person):
            palette = self.palette[person_idx]

            if 'silhouette' not in self.render_choice:
                if palette in PALETTE:
                    verts_rgb = PALETTE[palette][None, None].repeat(
                        num_frame, num_verts, 1)
                elif palette == 'random':
                    np.random.seed(person_idx)
                    nst0 = np.random.get_state()
                    verts_rgb = torch.FloatTensor(
                        np.random.uniform(low=0.5, high=1,
                                          size=(1, 1, 3))).repeat(
                                              num_frame, num_verts, 1)
                    np.random.set_state(nst0)
                elif palette == 'segmentation':
                    verts_labels = torch.zeros(num_verts)
                    cm = mpl_cm.get_cmap('jet')
                    norm_gt = mpl_colors.Normalize()
                    for part_idx, k in enumerate(
                            SEGMENTATION[self.model_type]['keys']):
                        index = SEGMENTATION[self.model_type]['func'](k)
                        verts_labels[index] = part_idx
                    verts_rgb = torch.ones(num_frame, num_verts, 3)
                    verts_rgb[..., :3] = torch.tensor(
                        cm(norm_gt(verts_labels))[..., :3])
            elif self.render_choice == 'silhouette':
                verts_rgb = PALETTE['white'][None, None].repeat(
                    num_frame, num_verts, 1)
            elif self.render_choice == 'part_silhouette':
                verts_rgb = torch.zeros(num_frame, num_verts, 1)
                for i, k in enumerate(self.slice_index):
                    verts_rgb[:, self.slice_index[k]] = 0.01 * (i + 1)
                Textures = TexturesCloset

            verts_pad.append(vertices[:,
                                      person_idx].view(num_frame, num_verts,
                                                       3))
            faces_pad.append(faces + int(num_verts * person_idx))
            verts_features_pad.append(verts_rgb)

        verts = torch.cat(verts_pad, 1)
        verts_features = torch.cat(verts_features_pad, 1)
        textures = Textures(verts_features=verts_features.to(self.device))
        faces = torch.cat(faces_pad, 1)
        meshes = Meshes(
            verts=verts.to(self.device),
            faces=faces.to(self.device),
            textures=textures)

        if camera_matrix is not None:
            camera_matrix = camera_matrix.to(self.device)

            if self.camera_type == 'weakperspective':
                camera_params = {
                    'scale_x': camera_matrix[:, 0],
                    'scale_y': camera_matrix[:, 1],
                    'trans_x': camera_matrix[:, 2],
                    'trans_y': camera_matrix[:, 3],
                    'trans_z': torch.max(verts[..., 2]),
                    'device': self.device,
                }
            else:
                R = camera_matrix[..., :3].T
                T = camera_matrix[..., 3].T
                camera_params = {
                    'device': self.device,
                    'R': R,
                    'T': T,
                    'zfar': 1000.0
                }
        else:
            elev = 0
            azim = 0
            R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
            camera_params = {
                'device': self.device,
                'R': R,
                'T': T,
                'zfar': 1000.0
            }
        cameras = self.camera_register(**camera_params)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=self.raster_settings),
            shader=self.shader(
                device=self.device,
                cameras=cameras,
                lights=self.lights,
                materials=self.materials,
                blend_params=self.blend_params) if
            (self.shader is not SoftSilhouetteShader) else self.shader())
        rendered_images = renderer(meshes)
        rgbs, valid_masks = rendered_images[
            ..., :3], (rendered_images[..., 3:] > 0) * 1.0
        if self.render_choice == 'part_silhouette':
            rendered_silhouettes = rgbs[None] * 100
            part_silhouettes = []
            for i in range(len(SEGMENTATION[self.model_type]['keys'])):
                part_silhouettes.append(1.0 *
                                        (rendered_silhouettes == (i + 1)) *
                                        rendered_silhouettes / (i + 1))
            part_silhouettes = torch.cat(part_silhouettes, 0)
            alphas = part_silhouettes[..., 0].permute(1, 2, 3, 0)
        elif self.render_choice == 'silhouette':
            alphas = rendered_images[..., 3] / (rendered_images[..., 3] + 1e-9)

        if self.obj_path and (self.render_choice != 'part_silhouette'):
            for index in range(num_frame):
                save_obj(
                    osp.join(self.obj_path,
                             Path(file_names[index]).stem + '.obj'),
                    vertices[index], faces[index])

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
                    output_images = output_images.detach().cpu().numpy(
                    ).astype(np.uint8)
                else:
                    output_images = (rgbs.detach().cpu().numpy() * 255).astype(
                        np.uint8)

            for index in range(output_images.shape[0]):
                folder = self.temp_path if self.temp_path is not None else\
                    self.output_path
                cv2.imwrite(
                    osp.join(folder, file_names[index]), output_images[index])

        if self.return_tensor:
            if 'silhouette' in self.render_choice:
                return alphas
            else:
                return rendered_images
        else:
            return None


class RenderDataset(Dataset):

    def __init__(self,
                 vertices: Union[np.ndarray, torch.Tensor],
                 cameras: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 file_format: str = '%06d.png',
                 images: Optional[Union[np.ndarray, torch.Tensor]] = None):
        super(RenderDataset, self).__init__()
        self.num_frames = vertices.shape[0]
        if images is not None:
            self.images = torch.from_numpy(images.astype(np.float32)) \
                if isinstance(images, np.ndarray) else images
            self.with_origin_image = True
        else:
            self.images = None
            self.with_origin_image = False
        self.vertices = torch.from_numpy(vertices.astype(
            np.float32)) if isinstance(vertices, np.ndarray) else vertices
        self.cameras = torch.from_numpy(cameras.astype(
            np.float32)) if isinstance(cameras, np.ndarray) else cameras
        self.len = self.num_frames
        self.file_format = file_format

    def __getitem__(self, index):
        result_dict = {
            'vertices': self.vertices[index],
            'file_names': self.file_format % (index),
        }
        if self.with_origin_image:
            result_dict.update({'images': self.images[index]})
        if self.cameras is not None:
            if self.cameras.shape[0] == self.num_frames:
                result_dict.update({'camera_matrix': self.cameras[index]})

            else:
                result_dict.update({'camera_matrix': self.cameras})
        return result_dict

    def __len__(self):
        return self.len
