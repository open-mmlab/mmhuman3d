# 相机

## 相机初始化

MMHuman3D遵循 `PyTorch3D` 中的相机规范：其外参矩阵定义为从相机坐标系到世界坐标系的变化。外参矩阵使用右乘，内参矩阵使用左乘。我们还提供了`OpenCV`中的相机规范，如果您更熟悉`OpenCV`，您也可以选择`OpenCV`规范来进行操作。

- **相机切片:**

    MMHuman3D中，推荐的相机初始化的方式为直接传递`K`, `R`, `T`矩阵。
    可以按索引对相机进行切片，也可以在batch维度进行拼接。

      ```python
      from mmhuman3d.core.cameras import PerspectiveCameras
      import torch
      K = torch.eye(4, 4)[None]
      R = torch.eye(3, 3)[None]
      T = torch.zeros(100, 3)
      # K, R, T的Batch大小应该相同，或者为1。如果不同，最终的batch大小会是三者中最大的那一个。
      cam = PerspectiveCameras(K=K, R=R, T=T)
      assert cam.R.shape == (100, 3, 3)
      assert cam.K.shape == (100, 4, 4)
      assert cam.T.shape == (100, 3)
      assert (cam[:10].K == cam.K[:10]).all()
      ```
- **构建相机:**

    封装在mmcv.Registry中。

    MMHuman3D中，推荐的初始化相机的方式为直接传递`K`, `R`, `T`矩阵, 但是也可以传递`focal_length` 和 `principle_point` 作为输入。

    以常用的`PerspectiveCameras`为例。如果`K`, `R`, `T`没有被确定，`K`将会使用`compute_default_projection_matrix`中默认的，`principal_point` 和 `R` 将会被设置为单位矩阵，`T`将会被设置为零矩阵。您也可以通过重写`compute_default_projection_matrix`来指定具体的数值。

    ```python
    from mmhuman3d.core.cameras import build_cameras

    # 使用给定的K, R, T矩阵初始化相机(PerspectiveCameras)。
    # K, R, T的batch大小应该相同，或者为1。
    K = torch.eye(4, 4)[None]
    R = torch.eye(3, 3)[None]
    T = torch.zeros(10, 3)

    height, width = 1000
    cam1 = build_cameras(
        dict(
            type='PerspectiveCameras',
            K=K,
            R=R,
            T=T,
            in_ndc=True,
            image_size=(height, width),
            convention='opencv',
            ))

    # This is the same as:
    cam2 = PerspectiveCameras(
            K=K,
            R=R,
            T=T,
            in_ndc=True,
            image_size=1000, # 图像尺度指定为单数时，表示正方形图像。
            convention='opencv',
            )
    assert cam1.K.shape == cam2.K.shape == (10, 4, 4)
    assert cam1.R.shape == cam2.R.shape == (10, 3, 3)
    assert cam1.T.shape == cam2.T.shape == (10, 3)

    # 使用给定的`image_size`, `principal_points`, `focal_length`初始化相机(PerspectiveCameras)。
    # `in_ndc = False` 意味着内参矩阵 `K` 在screen space中被定义。 `K`中的`focal_length`和`principal_point`被定义为像素个数。 下例中， `principal_points` 为 (500, 500) 像素， `focal_length` 为 1000 像素。
    cam = build_cameras(
        dict(
            type='PerspectiveCameras',
            in_ndc=False,
            image_size=(1000, 1000),
            principal_points=(500, 500),
            focal_length=1000,
            convention='opencv',
            ))

    assert (cam.K[0] == torch.Tensor([[1000., 0.,  500.,    0.],
                                      [0., 1000.,  500.,    0.],
                                      [0.,    0.,    0.,    1.],
                                      [0.,    0.,    1.,    0.]]).view(4, 4)).all()

    # 使用给定的K, R, T初始化相机(WeakPerspectiveCameras)。Weakperspective Camera 只支持`in_ndc = True`(默认)。
    cam = build_cameras(
        dict(
            type='WeakPerspectiveCameras',
            K=K,
            R=R,
            T=T,
            image_size=(1000, 1000)
            ))

    # 如果`K`, `R`, `T`矩阵没有给定，将会使用默认矩阵初始化`in_ndc`的相机(PerspectiveCameras)。
    cam = build_cameras(
        dict(
            type='PerspectiveCameras',
            in_ndc=True,
            image_size=(1000, 1000),
            ))
    # Then convert it to screen. This operation requires `image_size`.
    cam.to_screen_()
    ```

## 相机重投影矩阵
- **Perspective:**

    内参矩阵的格式:
    fx, fy 是焦距, px, py 是像主点。
    ```python
    K = [
            [fx,   0,   px,   0],
            [0,   fy,   py,   0],
            [0,    0,    0,   1],
            [0,    0,    1,   0],
        ]
    ```
    更多信息请参考[Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py#L895)。

- **WeakPerspective:**

    内参矩阵的格式:
    ```python
    K = [
            [sx*r,   0,    0,   tx*sx*r],
            [0,     sy,    0,   ty*sy],
            [0,     0,     1,       0],
            [0,     0,     0,       1],
        ]
    ```
    `WeakPerspectiveCameras` 事实上是正交投影, 主要用于 SMPL(x) 模型的重投影。
    更多信息请参考[mmhuman3d cameras](mmhuman3d/core/cameras/cameras.py#L40)。
    可以通过SMPL预测的相机参数进行转换:
    ```python
    from mmhuman3d.core.cameras import WeakPerspectiveCameras
    K = WeakPerspectiveCameras.convert_orig_cam_to_matrix(orig_cam)
    ```
    `pred_cam`是一个由[scale_x, scale_y, transl_x, transl_y]组成的array/tensor, 其维度为(frame, 4)。更多信息请参考[VIBE](https://github.com/mkocabas/VIBE/blob/master/lib/utils/renderer.py#L40-L47).

- **FoVPerspective:**

    内参矩阵的格式:
    ```python
    K = [
            [s1,   0,   w1,   0],
            [0,   s2,   h1,   0],
            [0,    0,   f1,  f2],
            [0,    0,    1,   0],
        ]
    ```
    s1, s2, w1, h1, f1, f2 由 FoV parameters (`fov`, `znear`, `zfar`, 等)定义, 更多信息请参考[Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py).

- **Orthographics:**

    内参矩阵的格式:
    ```python
    K = [
            [fx,   0,    0,  px],
            [0,   fy,    0,  py],
            [0,    0,    1,   0],
            [0,    0,    0,   1],
    ]
    ```
    更多信息请参考[Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py).

- **FoVOrthographics:**
    ```python
    K = [
            [scale_x,        0,         0,  -mid_x],
            [0,        scale_y,         0,  -mix_y],
            [0,              0,  -scale_z,  -mid_z],
            [0,              0,         0,       1],
    ]
    ```
    scale_x, scale_y, scale_z, mid_x, mid_y, mid_z 由 FoV parameters(`min_x`, `min_y`, `max_x`, `max_y`, `znear`, `zfar`, 等)定义, 更多信息请参考[Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py).

## 相机规范
- **在不同相机之间进行转换:**

    我们命名内参矩阵为`K`, 旋转矩阵为`R` 平移矩阵为`T`。
    不同的相机规范有着不同的轴向, 有些使用矩阵左乘，有些使用矩阵右乘。 内参矩阵和外参矩阵应该使用相同的矩阵乘法规范，但是`PyTorch3D`在计算过程中使用矩阵右乘，在相机初始化的时候传递左乘矩阵`K`(这主要是为了便于理解)。
    `NDC`和`screen`之间的转换也会影响内参矩阵, 这与相机规范无关。
    如果要使用现有的规范，请指定`['opengl', 'opencv', 'pytorch3d', 'pyrender', 'open3d']`中的一个。
    例如, 可以进行如下的操作，将使用`OpenCV`标定的相机转化为`PyTorch3D NDC`定义的相机以进行渲染:
    ```python
    from mmhuman3d.core.conventions.cameras import convert_cameras
    import torch

    K = torch.eye(4, 4)[None]
    R = torch.eye(3, 3)[None]
    T = torch.zeros(10, 3)
    height, width = 1080, 1920
    K, R, T = convert_cameras(
        K=K,
        R=R,
        T=T,
        in_ndc_src=False,
        in_ndc_dst=True,
        resolution_src=(height, width),
        convention_src='opencv',
        convention_dst='pytorch3d')
    ```
    输入 `K` 可以为 None, 或者维度为(batch_size, 3, 3) `array`/`tensor`，维度也可以为(batch_size, 4, 4)。

    输入 `R` 可以为 None, 或者维度为(batch_size, 3, 3) `array`/`tensor`。

    输入 `T` 可以为 None, 或者维度为(batch_size, 3) `array`/`tensor`。

    如果原始的 `K` 为 `None`, 将会一直保持为 `None`。如果原始的 `R` 为 `None`, 将会设置为单位矩阵。 如果原始的 `T` 为 `None`, 将会设置为零矩阵。

    更多关于`NDC`和`screen`相机空间的信息，请参考[Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md)。

- **定义个人的相机规范:**

    如果使用新的相机规范, 请在[CAMERA_CONVENTION_FACTORY](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/cameras/__init__.py)中, 按照向右, 向上, 和远离屏幕的顺序进行定义。例如, 下方第一个`PyRender`的顺序应该为'+x+y+z'。第二个`OpenCV`的顺序应该为'+x-y-z'。 第三个`Pytorch3D`的顺序为'-xyz'。
    ```
    OpenGL(PyRender)       OpenCV             Pytorch3D
        y                   z                     y
        |                  /                      |
        |                 /                       |
        |_______x        /________x     x________ |
        /                |                        /
       /                 |                       /
    z /                y |                    z /
    ```

## 部分转换函数
转换函数定义在`conventions.cameras`中。
- **NDC & screen:**

    ```python
    from mmhuman3d.core.conventions.cameras import (convert_ndc_to_screen,
                                                    convert_screen_to_ndc)

    K = convert_ndc_to_screen(K, resolution=(1080, 1920), is_perspective=True)
    K = convert_screen_to_ndc(K, resolution=(1080, 1920), is_perspective=True)
    ```

- **3x3 & 4x4 内参矩阵**

    ```python
    from mmhuman3d.core.conventions.cameras import (convert_K_3x3_to_4x4,
                                                    convert_K_4x4_to_3x3)

    K = convert_K_3x3_to_4x4(K, is_perspective=True)
    K = convert_K_4x4_to_3x3(K, is_perspective=True)
    ```

- **world & view:**

    在世界坐标和视图坐标中转换。

    ```python
    from mmhuman3d.core.conventions.cameras import convert_world_view
    R, T = convert_world_view(R, T)
    ```
- **weakperspective & perspective:**

    在weakperspective 和 perspective中进行转换,  `zmean` 参数是必须的。

    <!--弱透视相机是基于in_ndc的，如果透视关系ndc=False，必须传递分辨率参数。-->

    ```python
    from mmhuman3d.core.conventions.cameras import (
        convert_perspective_to_weakperspective,
        convert_weakperspective_to_perspective)

    K = convert_perspective_to_weakperspective(
        K, zmean, in_ndc=False, resolution, convention='opencv')
    K = convert_weakperspective_to_perspective(
        K, zmean, in_ndc=False, resolution, convention='pytorch3d')
    ```

## 部分计算函数

- **将3D坐标投影至平面:**

    ```python
    points_xydepth = cameras.transform_points_screen(points)
    points_xy = points_xydepth[..., :2]
    ```

- **计算点的深度:**

    可以简单地将点转化为视图坐标，并且得到深度z。
    更多的用例请参考[DepthRenderer](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/visualization/renderer/torch3d_renderer/depth_renderer.py).
    ```python
    points_depth = cameras.compute_depth_of_points(points)
    ```

- **获取mesh的法线:**

    使用`Pytorch3D`计算mesh的法线。
    更多的用例请参考[NormalRenderer](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/visualization/renderer/torch3d_renderer/normal_renderer.py).
    ```python
    normals = cameras.compute_normal_of_meshes(meshes)
    ```

- **获取相机平面的法线:**

    获取从相机中心指向相机平面法线的归一化张量。
    ```python
    normals = cameras.get_camera_plane_normals()
    ```
