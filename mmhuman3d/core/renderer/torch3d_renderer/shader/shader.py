from typing import Optional

import torch
import torch.nn as nn
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import BlendParams, hard_rgb_blend
from pytorch3d.renderer.mesh.shader import SoftSilhouetteShader
from pytorch3d.structures.utils import padded_to_packed


class SilhouetteShader(SoftSilhouetteShader):
    """Avoid unexpected keyword argument error."""

    def __init__(self,
                 blend_params: Optional[BlendParams] = None,
                 **kwargs) -> None:
        super().__init__(blend_params)


class NoLightShader(nn.Module):
    """No light shader."""

    def __init__(self,
                 blend_params: Optional[BlendParams] = None,
                 **kwargs) -> None:
        """Initlialize without blend_params."""
        super().__init__()
        self.blend_params = blend_params if blend_params is not None\
            else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        """Sample without light."""
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get('blend_params', self.blend_params)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images


class DepthShader(nn.Module):
    """No light shader."""

    def __init__(self,
                 blend_params: Optional[BlendParams] = None,
                 **kwargs) -> None:
        """Initlialize without blend_params."""
        super().__init__()
        self.blend_params = blend_params if blend_params is not None\
            else BlendParams()

    def forward(self, fragments, meshes, cameras, **kwargs) -> torch.Tensor:
        """Sample without light."""
        verts_depth = cameras.compute_depth_of_points(meshes.verts_padded())
        faces = meshes.faces_packed()  # (F, 3)
        verts_depth = padded_to_packed(verts_depth)
        faces_depth = verts_depth[faces]
        depth_map = interpolate_face_attributes(
            pix_to_face=fragments.pix_to_face,
            barycentric_coords=fragments.bary_coords,
            face_attributes=faces_depth)
        return depth_map[..., 0, :]


class NormalShader(nn.Module):
    """No light shader."""

    def __init__(self,
                 blend_params: Optional[BlendParams] = None,
                 **kwargs) -> None:
        """Initlialize without blend_params."""
        super().__init__()
        self.blend_params = blend_params if blend_params is not None\
            else BlendParams()

    def forward(self, fragments, meshes, cameras, **kwargs) -> torch.Tensor:
        """Sample without light."""
        verts_normal = cameras.compute_normal_of_meshes(meshes)
        faces = meshes.faces_packed()  # (F, 3)
        verts_normal = padded_to_packed(verts_normal)
        faces_normal = verts_normal[faces]
        normal_map = interpolate_face_attributes(
            pix_to_face=fragments.pix_to_face,
            barycentric_coords=fragments.bary_coords,
            face_attributes=faces_normal)
        return normal_map[..., 0, :]


class SegmentationShader(nn.Module):
    """No light shader."""

    def __init__(self,
                 blend_params: Optional[BlendParams] = None,
                 **kwargs) -> None:
        """Initlialize without blend_params."""
        super().__init__()
        self.blend_params = blend_params if blend_params is not None\
            else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        """Sample without light."""
        verts_class = meshes.textures.verts_features_padded()
        faces = meshes.faces_packed()  # (F, 3)
        verts_class = padded_to_packed(verts_class)
        faces_class = verts_class[faces]
        segmentation_map = interpolate_face_attributes(
            pix_to_face=fragments.pix_to_face,
            barycentric_coords=fragments.bary_coords,
            face_attributes=faces_class).long()
        return segmentation_map[..., :, 0]
