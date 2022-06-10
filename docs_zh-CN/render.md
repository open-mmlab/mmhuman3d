# 渲染 Meshes

## 渲染器初始化

我们使用 `Pytorch3D` 渲染器。通过光栅化器、着色器和其他设定初始化渲染器。 `MMHuman3D`中的渲染器兼容`Pytorch3D`的渲染器，但功能更加丰富且更易用。 例如，您可以像`Pytorch3D`那样，通过传入光栅化器和着色器初始化渲染器，也可以传入包含相关设置的字典或使用默认的设置。
在`mmhuman3d`中, 我们提供了不同的渲染器，包括 `MeshRenderer`, `DepthRenderer`, `NormalRenderer`, `PointCloudRenderer`, `SegmentationRenderer`, `SilhouetteRenderer` 和 `UVRenderer`。其中, `UVRenderer` 比较特别，请参考[UVRenderer](#uvrenderer)。

上述所有的渲染器可以通过`MMCV.Registry`进行初始化。通过字典储存渲染器的配置会很方便。

- **`pytorch3d`和`mmhuman3d`的比较:**
```python
### 通过Pytorch3D初始化
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

### 通过mmhuman3d初始化
from mmhuman3d.core.visualization.renderer import MeshRenderer
# 可以通过nn.Module和dict初始化光栅化器
rasterizer = dict(
    image_size=128,
    blur_radius=0.0,
    faces_per_pixel=1,
)
# 可以通过nn.Module和dict初始化光线
lights = dict(type='point', ambient_color=((0.5, 0.5, 0.5), ),
    diffuse_color=((0.3, 0.3, 0.3), ),
    specular_color=((0.2, 0.2, 0.2), ),
    direction=((0, 1, 0), ),)

# 可以通过camera和字典初始化光栅化器
cameras = dict(type='fovperspective', R=R, T=T, device=device)

# 可以通过nn.Module和dict初始化着色器
shader = dict(type='SoftPhongShader')
```

这两种方式是等价的。
```python
import torch.nn as nn
from mmhuman3d.core.visualization.renderer import MeshRenderer, build_renderer

renderer = MeshRenderer(shader=shader, device=device, rasterizer=rasterizer, resolution=resolution)
renderer = build_renderer(dict(type='mesh', device=device, shader=shader, rasterizer=rasterizer, resolution=resolution))

# 使用默认的光栅化器和着色器的配置
renderer = build_renderer(dict(type='mesh', device=device, resolution=resolution))
assert isinstance(renderer.rasterizer, nn.Module)
assert isinstance(renderer.shader, nn.Module)
```

我们提供了用于可视化的`tensor2rgba`函数，其返回的是可视化彩色图像的张量。
对于不同的渲染器，该函数是不同的。例如，`DepthRenderer`渲染出的是一个形状为(N, H, W, 1)的张量，我们将其复制成形状为(N, H, W, 4)的图像张量。`SegmentationRenderer`渲染出的是一个形状为(N, H, W, C)的张量，我们根据色彩匹配关系将其转换成(N, H, W, 4)的图像张量。`NormalRenderer`渲染出的是一个形状为(N, H, W, 4)的张量，其取值范围为[-1, 1]，`tensor2rgba`会将其归一化到[0, 1]。

```python
import torch
from mmhuman3d.core.visualization.renderer import build_renderer

renderer = build_renderer(dict(type='mesh', device=device, resolution=resolution))
rendered_tensor = renderer(meshes=meshes, cameras=cameras, lights=lights)
rendered_rgba = renderer.tensor2rgba(rendered_tensor)
```

`MMHuman3D`中的渲染器可以更改输出设置，并且提供了文件的输入输出操作，`tensor2rgba`函数会对写入的图像和视频进行转换。

```python
# 写入视频
renderer = build_renderer(dict(type='mesh', device=device, resolution=resolution, output_path='test.mp4'))
backgrounds = torch.Tensor(N, H, W, 3)
rendered_tensor = renderer(meshes=meshes, cameras=cameras, lights=lights, backgrounds=backgrounds)
renderer.export() #这一步对于视频是必要的

# 在文件夹中写入图像
renderer = build_renderer(dict(type='mesh', device=device, resolution=resolution, output_path='test_folder', out_img_format='%06d.png'))
backgrounds = torch.Tensor(N, H, W, 3)
rendered_tensor = renderer(meshes=meshes, cameras=cameras, lights=lights, backgrounds=backgrounds)
```


## 使用 render_runner

您可以通过`render_runner`传入用于渲染的数据。它会使用for循环批量渲染张量，因此可以对一长串视频序列进行渲染，而不会出现`CUDA out of memory`的报错。

```python
import torch
from mmhuman3d.core.visualization.renderer import render_runner

render_data = dict(cameras=cameras, lights=lights, meshes=meshes, backgrounds=backgrounds)
# 对于不可微的渲染，指定no_grad=True
rendered_tensor = render_runner.render(renderer=renderer, output_path=output_path, resolution=resolution, batch_size=batch_size, device=device, no_grad=True, return_tensor=True, **render_data)
```

## UVRenderer

`UVRenderer` 不同于其他的渲染器，它是一个smpl uv拓扑定义的包装器和采样器。其包含有两个功能: 将顶点的属性包装为map, 从map上采样顶点属性。

### 初始化
UV 信息储存在`smpl_uv.npz`文件中。
```python
uv_renderer = build_renderer(dict(type='uv', resolution=resolution, device=device, model_type='smpl', uv_param_path='data/body_models/smpl/smpl_uv.npz'))
```
### 包装
将灰色的纹理图像变形为smpl的mesh。

```python
import torch
from mmhuman3d.models import build_body_model
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
### 采样
从map采样顶点。

```python
normal_map = torch.ones(1, 512, 512, 3)
vertex_normals = uv_renderer.vertex_resample(normal_map)
assert vertex_normals.shape == (1 ,6890, 3)
assert (vertex_normals == 1).all()
```
