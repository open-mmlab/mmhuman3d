from typing import Union

import torch
from pytorch3d.renderer.lighting import AmbientLights as _AmbientLights
from pytorch3d.renderer.lighting import DirectionalLights as _DirectionalLights
from pytorch3d.renderer.lighting import PointLights as _PointLights
from pytorch3d.renderer.utils import TensorProperties

MMLIGHT_ATTR = [
    'ambient_color', 'diffuse_color', 'specular_color', 'location', 'direction'
]


class MMLights(TensorProperties):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        _N = 1
        self.mmlight_attr_list = []
        for attr_name in MMLIGHT_ATTR:
            if hasattr(self, attr_name):
                self.mmlight_attr_list.append(attr_name)
        for k in self.mmlight_attr_list:
            v = getattr(self, k)
            if not isinstance(v, torch.Tensor):
                v = torch.Tensor(v)
            v = v.view(-1, 3)
            setattr(self, k, v)

            if getattr(self, k).shape[0] > _N:
                _N = getattr(self, k).shape[0]
        for k in self.mmlight_attr_list:
            if getattr(self, k).shape[0] == 1:
                setattr(self, k, getattr(self, k).repeat(_N, 1))
        self._N = _N

    def __len__(self, ):
        return self._N

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, int):
            index = [index]
        kwargs = {}
        for k in self.mmlight_attr_list:
            kwargs[k] = getattr(self, k)[index]

        return self.__class__(device=self.device, **kwargs)

    def extend(self, N):
        kwargs = {}
        for k in self.mmlight_attr_list:
            kwargs[k] = getattr(self, k).repeat(N, 1)
        return self.__class__(device=self.device, **kwargs)

    def extend_(self, N):
        for k in self.mmlight_attr_list:
            setattr(self, k, getattr(self, k).repeat(N, 1))
        self._N = N


class AmbientLights(_AmbientLights, MMLights):

    def __init__(self, ambient_color=None, device='cpu', **kwargs) -> None:
        if ambient_color is None:
            ambient_color = ((1.0, 1.0, 1.0), )
        diffuse_color = ((0.0, 0.0, 0.0), )
        super(_AmbientLights, self).__init__(
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            device=device)

    def __getitem__(self, index: Union[int, slice]):
        return super(_AmbientLights, self).__getitem__(index)


class PointLights(_PointLights, MMLights):

    def __getitem__(self, index: Union[int, slice]):
        return super(_PointLights, self).__getitem__(index)


class DirectionalLights(_DirectionalLights, MMLights):

    def __getitem__(self, index: Union[int, slice]):
        return super(_DirectionalLights, self).__getitem__(index)
