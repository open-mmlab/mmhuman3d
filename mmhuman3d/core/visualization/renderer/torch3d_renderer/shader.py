from typing import Optional, Union

import torch
import torch.nn as nn
from pytorch3d.renderer import BlendParams, hard_rgb_blend


class NoLightShader(nn.Module):

    def __init__(self,
                 device: Union[torch.device, str] = 'cpu',
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
