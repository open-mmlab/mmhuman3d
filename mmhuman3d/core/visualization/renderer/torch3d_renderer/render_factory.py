import torch
from pytorch3d.renderer import (
    DirectionalLights,
    HardFlatShader,
    PointLights,
    SoftGouraudShader,
    SoftPhongShader,
    SoftSilhouetteShader,
)
from pytorch3d.renderer.cameras import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
)

from mmhuman3d.core.cameras import WeakPerspectiveCamerasVibe
from mmhuman3d.core.conventions.segmentation.smpl import (
    SMPL_SEGMENTATION_DICT,
    smpl_part_segmentation,
)
from mmhuman3d.core.conventions.segmentation.smplx import (
    SMPLX_SEGMENTATION_DICT,
    smplx_part_segmentation,
)
from .shader import NoLightShader

SHADER_FACTORY = {
    'phong': SoftPhongShader,
    'gouraud': SoftGouraudShader,
    'silhouette': SoftSilhouetteShader,
    'flat': HardFlatShader,
    'nolight': NoLightShader,
}

CAMERA_FACTORY = {
    'perspective': PerspectiveCameras,
    'orthographic': OrthographicCameras,
    'fovperspective': FoVPerspectiveCameras,
    'fovorthographic': FoVOrthographicCameras,
    'weakperspective': WeakPerspectiveCamerasVibe,
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

SMPL_SEGMENTATION = {
    'smpl': {
        'keys': SMPL_SEGMENTATION_DICT.keys(),
        'func': smpl_part_segmentation
    },
    'smplx': {
        'keys': SMPLX_SEGMENTATION_DICT.keys(),
        'func': smplx_part_segmentation
    }
}
