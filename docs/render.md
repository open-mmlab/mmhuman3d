# Render Meshes

## Renderer Initialization

We follow `Pytorch3D` renderer. We initialize the renderer with rasterizer, shader and other settings. Ours is compatible with `Pytorch3D` renderer initializations, but more flexible and functional. E.g., you can initialize a renderer just like `Pytorch3D` by passing the rasterizer and shader modules, or you can pass setting `dicts`, or use default settings.
In `mmhuman3d`, we provide `MeshRenderer`, `DepthRenderer`, `NormalRenderer`, `PointCloudRenderer`, `SegmentationRenderer`, `SilhouetteRenderer` and `UVRenderer`. In these renderers, `UVRenderer` is special and please refer to the last chapter [UVRenderer](#uvrenderer).

All of these renderers could be initialized by MMCV.Registry. It is convenient to store the renderers configs by dicts.

- **comparison between `pytorch3d` and `mmhuman3d`:**
```python
### initialized by Pytorch3D
import torch
from pytorch3d.renderer import MeshRenderer, RasterizationSettings
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.cameras import FoVPerspectiveCameras

device = torch.device('cuda')
R, T = look_at_view_transform(dist=2.7, elev=0, azim=0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

lights = PointLights(
    device=device,
    ambient_color=((0.5, 0.5, 0.5), ),
    diffuse_color=((0.3, 0.3, 0.3), ),
    specular_color=((0.2, 0.2, 0.2), ),
    direction=((0, 1, 0), ),
)
raster_settings = RasterizationSettings(
    image_size=128,
    blur_radius=0.0,
    faces_per_pixel=1,
)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights))

### initialized by mmhuman3d
from mmhuman3d.core.visualization.renderer import MeshRenderer
# rasterizer could be passed by nn.Module or dict
rasterizer = dict(
    image_size=128,
    blur_radius=0.0,
    faces_per_pixel=1,
)
# lights could be passed by nn.Module or dict
lights = dict(type='point', ambient_color=((0.5, 0.5, 0.5), ),
    diffuse_color=((0.3, 0.3, 0.3), ),
    specular_color=((0.2, 0.2, 0.2), ),
    direction=((0, 1, 0), ),)

# rasterizer could be passed by cameras or dict
cameras = dict(type='fovperspective', R=R, T=T, device=device)

# shader could be passed by nn.Module or dict
shader = dict(type='SoftPhongShader')

# These two methods are equal.
renderer = MeshRenderer(shader=shader, device=device, rasterizer=rasterizer, resolution=resolution)
renderer = build_renderer(dict(type='mesh', device=device, shader=shader, rasterizer=rasterizer, resolution=resolution))

# Use default raster and shader settings
renderer = build_renderer(dict(type='mesh', device=device, resolution=resolution))
```

 Moreover, our renderer could set output settings and provide file I/O operations.

 ```python
 ```

We provide `tensor2rgba` function for visualization, the returned tensor will be a colorful image for visualization.

 ```python
 ```

 We provide different type of renderers with different forward functions and visualization functions.

 ```python
 ```


## Run Renderers

You could pass your renderer, cameras, lights by our `render_runner` to run a series batch of render. You could also use `DataParallel` for the renderers to run multi-GPU render.


## Explanation for the rendered results

### MeshRenderer

### DepthRenderer

### NormalRenderer

### PointCloudRenderer

### SegmentationRenderer

### SilhouetteRenderer


## UVRenderer

Our `UVRenderer` is different from the above renderers. It is actually a smpl uv topology defined wrapper and sampler. It has two main functions: wrapping vertex attributes to a map, sampling vertex attributes from a map.

#### warping

#### sampling
