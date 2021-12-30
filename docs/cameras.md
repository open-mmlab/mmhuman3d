# Cameras

## Camera Initialization

We follow `Pytorch3D` cameras. The camera extrinsic matrix is defined as the camera to world transformation, and uses right matrix multiplication, whereas the intrinsic matrix uses left matrix multiplication. Nevertheless, our interface provides `opencv` convention that defines the camera the same way as an `OpenCV` camera, would be helpful if you are more familiar with that.

- **Slice cameras:**

    In mmhuman3d, the recommended way to initialize a camera is by passing `K`, `R`, `T` matrix directly.
    You can slice the cameras by index. You can also concat the cameras in batch dim.

    ```python
    from mmhuman3d.core.cameras import PerspectiveCameras
    import torch
    K = torch.eye(4, 4)[None]
    R = torch.eye(3, 3)[None]
    T = torch.zeros(100, 3)
    # Batch of K, R, T should all be the same or some of them could be 1. The final batch size will be the biggest one.
    cam = PerspectiveCameras(K=K, R=R, T=T)
    assert cam.R.shape == (100, 3, 3)
    assert cam.K.shape == (100, 4, 4)
    assert cam.T.shape == (100, 3)
    assert (cam[:10].K == cam.K[:10]).all()
    ```
- **Build cameras:**

    Wrapped by mmcv.Registry.
    In mmhuman3d, the recommended way to initialize a camera is by passing `K`, `R`, `T` matrix directly, but you also have the options to pass `focal_length` and `principle_point` as the input.

    Take the usually used `PerspectiveCameras` as examples. If `K`, `R`, `T` are not specified, the `K` will use default `K` by `compute_default_projection_matrix` with default `focal_length` and `principal_point` and `R` will be identical matrix, `T` will be zeros. You can also specify by overwriting the parameters for `compute_default_projection_matrix`.
    ```python
    from mmhuman3d.core.cameras import build_cameras

    # Initialize a perspective camera with given K, R, T matrix.
    # It is recommended that the batches of K, R, T either the same or be 1.
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
            image_size=1000, # single number represents square images.
            convention='opencv',
            )
    assert cam1.K.shape == cam2.K.shape == (10, 4, 4)
    assert cam1.R.shape == cam2.R.shape == (10, 3, 3)
    assert cam1.T.shape == cam2.T.shape == (10, 3)

    # Initialize a perspective camera with specific `image_size`, `principal_points`, `focal_length`.
    # `in_ndc = False` means the intrinsic matrix `K` defined in screen space. The `focal_length` and `principal_point` in `K` is defined in scale of pixels. This `principal_points` is (500, 500) pixels and `focal_length` is 1000 pixels.
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

    # Initialize a weakperspective camera with given K, R, T. weakperspective camera support `in_ndc = True` only.
    cam = build_cameras(
        dict(
            type='WeakPerspectiveCameras',
            K=K,
            R=R,
            T=T,
            image_size=(1000, 1000)
            ))

    # If no `K`, `R`, `T` information provided
    # Initialize a `in_ndc` perspective camera with default matrix.
    cam = build_cameras(
        dict(
            type='PerspectiveCameras',
            in_ndc=True,
            image_size=(1000, 1000),
            ))
    # Then convert it to screen. This operation requires `image_size`.
    cam.to_screen_()
    ```

## Camera Projection Matrixs
- **Perspective:**

    format of intrinsic matrix:
    fx, fy is focal_length, px, py is principal_point.
    ```python
    K = [
            [fx,   0,   px,   0],
            [0,   fy,   py,   0],
            [0,    0,    0,   1],
            [0,    0,    1,   0],
        ]
    ```
    Detailed information refer to [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py#L895).

- **WeakPerspective:**

    format of intrinsic matrix:
    ```python
    K = [
            [sx*r,   0,    0,   tx*sx*r],
            [0,     sy,    0,   ty*sy],
            [0,     0,     1,       0],
            [0,     0,     0,       1],
        ]
    ```
    `WeakPerspectiveCameras` is orthographics indeed, mainly for SMPL(x) projection.
    Detailed information refer to [mmhuman3d cameras](mmhuman3d/core/cameras/cameras.py#L40).
    This can be converted from SMPL predicted camera parameter by:
    ```python
    from mmhuman3d.core.cameras import WeakPerspectiveCameras
    K = WeakPerspectiveCameras.convert_orig_cam_to_matrix(orig_cam)
    ```
    The pred_cam is array/tensor of shape (frame, 4) consists of [scale_x, scale_y, transl_x, transl_y]. See in [VIBE](https://github.com/mkocabas/VIBE/blob/master/lib/utils/renderer.py#L40-L47).

- **FoVPerspective:**
    ```python
    format of intrinsic matrix:
    K = [
            [s1,   0,   w1,   0],
            [0,   s2,   h1,   0],
            [0,    0,   f1,  f2],
            [0,    0,    1,   0],
        ]
    ```
    s1, s2, w1, h1, f1, f2 are defined by FoV parameters (`fov`, `znear`, `zfar`, etc.), detailed information refer to [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py).

- **Orthographics:**

    format of intrinsic matrix:
    ```python
    K = [
            [fx,   0,    0,  px],
            [0,   fy,    0,  py],
            [0,    0,    1,   0],
            [0,    0,    0,   1],
    ]
    ```
    Detailed information refer to [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py).

- **FoVOrthographics:**
    ```python
    K = [
            [scale_x,        0,         0,  -mid_x],
            [0,        scale_y,         0,  -mix_y],
            [0,              0,  -scale_z,  -mid_z],
            [0,              0,         0,       1],
    ]
    ```
    scale_x, scale_y, scale_z, mid_x, mid_y, mid_z are defined by FoV parameters(`min_x`, `min_y`, `max_x`, `max_y`, `znear`, `zfar`, etc.), related information refer to [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py).

## Camera Conventions
- **Convert between different cameras:**

    We name intrinsic matrix as `K`, rotation matrix as `R` and translation matrix as `T`.
    Different camera conventions have different axis directions, and some use left matrix multiplication and some use right matrix multiplication. Intrinsic and extrinsic matrix should be of the same multiplication convention, but some conventions like `Pytorch3D` uses right matrix multiplication in computation procedure but passes left matrix multiplication `K` when initializing the cameras(mainly for better understanding).
    Conversion between `NDC` (normalized device coordinate) and `screen` also influence the intrinsic matrix, this is independent of camera conventions but should also be included.
    If you want to use an existing convention, choose in `['opengl', 'opencv', 'pytorch3d', 'pyrender', 'open3d']`.
    E.g., you want to convert your opencv calibrated camera to Pytorch3D NDC defined camera for rendering, you can do:
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
    Input K could be None, or `array`/`tensor` of shape (batch_size, 3, 3) or (batch_size, 4, 4).
    Input R could be None, or `array`/`tensor` of shape (batch_size, 3, 3).
    Input T could be None, or `array`/`tensor` of shape (batch_size, 3).
    If the original `K` is `None`, it will remain `None`. If the original `R` is `None`, it will be set as identity matrix. If the original `T` is `None`, it will be set as zeros matrix.
    Please refer to [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md) for more information about cameras in `NDC` and in `screen` space..

- **Define your new camera convention:**

    If want to use a new convention, define your convention in [CAMERA_CONVENTION_FACTORY](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/cameras/__init__.py) by the order of right to, up to, and off screen. E.g., the first one is pyrender and its convention should be '+x+y+z'. '+' could be ignored. The second one is opencv and its convention should be '+x-y-z'. The third one is Pytorch3D and its convention should be '-xyz'.
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

## Some Conversion Functions
Convert functions are also defined in conventions.cameras.
- **NDC & screen:**

    ```python
    from mmhuman3d.core.conventions.cameras import (convert_ndc_to_screen,
                                                    convert_screen_to_ndc)

    K = convert_ndc_to_screen(K, resolution=(1080, 1920), is_perspective=True)
    K = convert_screen_to_ndc(K, resolution=(1080, 1920), is_perspective=True)
    ```

- **3x3 & 4x4 intrinsic matrix**

    ```python
    from mmhuman3d.core.conventions.cameras import (convert_K_3x3_to_4x4,
                                                    convert_K_4x4_to_3x3)

    K = convert_K_3x3_to_4x4(K, is_perspective=True)
    K = convert_K_4x4_to_3x3(K, is_perspective=True)
    ```

- **world & view:**

    Convert between world & view coordinates.

    ```python
    from mmhuman3d.core.conventions.cameras import convert_world_view
    R, T = convert_world_view(R, T)
    ```
- **weakperspective & perspective:**

    Convert between weakperspective & perspective. zmean is needed.
    WeakperspectiveCameras is in_ndc, so you should pass resolution if perspective not in ndc.

    ```python
    from mmhuman3d.core.conventions.cameras import (
        convert_perspective_to_weakperspective,
        convert_weakperspective_to_perspective)

    K = convert_perspective_to_weakperspective(
        K, zmean, in_ndc=False, resolution, convention='opencv')
    K = convert_weakperspective_to_perspective(
        K, zmean, in_ndc=False, resolution, convention='pytorch3d')
    ```

## Some Compute Functions

- **Project 3D coordinates to screen:**

    ```python
    points_xydepth = cameras.transform_points_screen(points)
    points_xy = points_xydepth[..., :2]
    ```

- **Compute depth of points:**

    You can simply convert points to the view coordinates and get the z value as depth. Example could be found in [DepthRenderer](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/visualization/renderer/torch3d_renderer/depth_renderer.py).
    ```python
    points_depth = cameras.compute_depth_of_points(points)
    ```

- **Compute normal of meshes:**

    Use `Pytorch3D` to compute normal of meshes. Example could be found in [NormalRenderer](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/visualization/renderer/torch3d_renderer/normal_renderer.py).
    ```python
    normals = cameras.compute_normal_of_meshes(meshes)
    ```

- **Get camera plane normal:**

    Get the normalized normal tensor which points out of the camera plane from camera center.
    ```python
    normals = cameras.get_camera_plane_normals()
    ```
