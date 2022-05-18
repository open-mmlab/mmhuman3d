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
```

These two methods are equal.
```python
import torch.nn as nn
from mmhuman3d.core.visualization.renderer import MeshRenderer, build_renderer

renderer = MeshRenderer(shader=shader, device=device, rasterizer=rasterizer, resolution=resolution)
renderer = build_renderer(dict(type='mesh', device=device, shader=shader, rasterizer=rasterizer, resolution=resolution))

# Use default raster and shader settings
renderer = build_renderer(dict(type='mesh', device=device, resolution=resolution))
assert isinstance(renderer.rasterizer, nn.Module)
assert isinstance(renderer.shader, nn.Module)
```

We provide `tensor2rgba` function for visualization, the returned tensor will be a colorful image for visualization.
 This function is different for different renderers. E.g., the rendered tensor of `DepthRenderer` is shape of (N, H, W, 1) of depth, and we will repeat it as a (N, H, W, 4) image tensor. And the rendered tensor of `SegmentationRenderer` is shape of (N, H, W, C) LongTensor, and we will convert it as a (N, H, W, 4) colorful image tensor according to a colormap. The rendered tensor of `NormalRenderer` is a (N, H, W, 4), its range is [-1, 1] and the `tensor2rgba` will normalize it to [0, 1].

 The operation is simple:
 ```python
 import torch
 from mmhuman3d.core.visualization.renderer import build_renderer

 renderer = build_renderer(dict(type='mesh', device=device, resolution=resolution))
 rendered_tensor = renderer(meshes=meshes, cameras=cameras, lights=lights)
 rendered_rgba = renderer.tensor2rgba(rendered_tensor)
 ```

 Moreover, our renderer could set output settings and provide file I/O operations.
 These writed images or videos are converted by the mentioned function `tensor2rgba`.

 ```python
 # will write a video
 renderer = build_renderer(dict(type='mesh', device=device, resolution=resolution, output_path='test.mp4'))
 backgrounds = torch.Tensor(N, H, W, 3)
 rendered_tensor = renderer(meshes=meshes, cameras=cameras, lights=lights, backgrounds=backgrounds)
 renderer.export() # needed for a video

 # will write a folder of images
 renderer = build_renderer(dict(type='mesh', device=device, resolution=resolution, output_path='test_folder', out_img_format='%06d.png'))
 backgrounds = torch.Tensor(N, H, W, 3)
 rendered_tensor = renderer(meshes=meshes, cameras=cameras, lights=lights, backgrounds=backgrounds)
 ```


## Use render_runner

You could pass your data by `render_runner` to render a series batch of render. It will use a for loop to render the tensor by batch so you can render a long sequence of video without `CUDA out of memory`.

```python
import torch
from mmhuman3d.core.visualization.renderer import render_runner

render_data = dict(cameras=cameras, lights=lights, meshes=meshes, backgrounds=backgrounds)
# no_grad=True for non-differentiable render
rendered_tensor = render_runner.render(renderer=renderer, output_path=output_path, resolution=resolution, batch_size=batch_size, device=device, no_grad=True, return_tensor=True, **render_data)
```

## UVRenderer

Our `UVRenderer` is different from the above renderers. It is actually a smpl uv topology defined wrapper and sampler. It has two main utilities: wrapping vertex attributes to a map, sampling vertex attributes from a map.

### Initialize
The UV information is stored in the `smpl_uv.npz` file.
```python
uv_renderer = build_renderer(dict(type='uv', resolution=resolution, device=device, model_type='smpl', uv_param_path='data/body_models/smpl/smpl_uv.npz'))
```
### warping
Warp a gray texture image to smpl_mesh.

```python
import torch
from mmhuman3d.models.body_models.builder import build_body_model
from pytorch3d.structures import meshes
from mmhuman3d.core.visualization.renderer import build_renderer
body_model = build_body_model(dict(type='smpl', model_path=model_path)).to(device)
pose_dict = body_model.tensor2dict(torch.zeros(1, 72))
verts = body_model(**pose_dict)['vertices']
faces = body_model.faces_tensor[None]
smpl_mesh = Meshes(verts=verts, faces=faces)
texture_image = torch.ones(1, 512, 512, 3) * 0.5
smpl_mesh.textures = uv_renderer.warp_texture(texture_image=texture_image)
```
### sampling
Sample vertex normal from a normal_map.

```python
normal_map = torch.ones(1, 512, 512, 3)
vertex_normals = uv_renderer.vertex_resample(normal_map)
assert vertex_normals.shape == (1 ,6890, 3)
assert (vertex_normals == 1).all()
```
