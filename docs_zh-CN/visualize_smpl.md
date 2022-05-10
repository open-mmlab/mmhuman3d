## 可视化`SMPL`的Mesh
- **快速可视化没有背景的图像的`smpl(x)`姿态:**

    假定smpl姿态参数为形状为(frame, 165)的张量或数组
    ```python
    from mmhuman3d.core.visualization import visualize_smpl_pose
    body_model_config = dict(
        type='smpl', model_path=model_path)
    visualize_smpl_pose(
        poses=poses,
        output_path='smpl.mp4',
        resolution=(1024, 1024))
    ```

    或者smplx姿态参数为形状为(frame, 165)的张量或数组
    ```python
    body_model_config = dict(
        type='smplx', model_path=model_path)
    visualize_smpl_pose(
        poses=poses,
        body_model_config=body_model_config,
        output_path='smplx.mp4',
        resolution=(1024, 1024))
    ```
    您也可以输入遵循smplx定义的字典张量。可以参考 [visualize_smpl](mmhuman3d/core/visualization/visualize_smpl.py#L166-211) 和 [original smplx](https://github.com/vchoutas/smplx/blob/master/smplx/body_models.py)。


- **可视化T-pose:**
    如果想可视化一个T-pose的smpl模型，或者没有全局旋转信息，您可以采取如下操作:
    ```python
    import torch
    from mmhuman3d.core.visualization import visualize_T_pose
    body_model_config = dict(
        type='smpl', model_path=model_path)
    visualize_T_pose(
        num_frames=100,
        body_model_config=body_model_config,
        output_path='smpl_tpose.mp4',
        orbit_speed=(1, 0.5),
        resolution=(1024, 1024))
    ```

- **使用预测的VIBE相机可视化smpl模型:**
    假定您有以下三种参数: 形状为(frame, 72)的`poses`(numpy/tensor), 形状为(frame, 10)的`betas`, 形状为(10, 4)的`pred_cam`。
    以vibe的sample_video.mp4为例:
    ```python
    import pickle
    from mmhuman3d.core.visualization import visualize_smpl_vibe
    with open('vibe_output.pkl', 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    poses = d[1]['pose']
    orig_cam = d[1]['orig_cam']
    pred_cam = d[1]['pred_cam']
    bbox = d[1]['bboxes']
    gender = 'female'

    # 传入 pred_cam 和 bbox
    body_model_config = dict(
        type='smpl', model_path=model_path, gender=gender)
    visualize_smpl_vibe(
        poses=poses,
        betas=betas,
        body_model_config=body_model_config,
        pred_cam=pred_cam,
        bbox=bbox,
        output_path='vibe_demo.mp4',
        origin_frames='sample_video.mp4',
        resolution=(1024, 1024))

    # 或者传入 orig_cam
    body_model_config = dict(
        type='smpl', model_path=model_path, gender=gender)
    visualize_smpl_vibe(
        poses=poses,
        betas=betas,
        body_model_config=body_model_config,
        orig_cam=orig_cam,
        output_path='vibe_demo.mp4',
        origin_frames='sample_video.mp4',
        resolution=(1024, 1024))

    ```

- **使用预测的HMR/SPIN相机可视化smpl模型:**
    假定您有以下三种参数: 形状为(frame, 72)的`poses`(numpy/tensor), 形状为(frame, 10)的`betas`, 形状为(10, 4)的`cam_translation`。
    ```python
    import pickle
    from mmhuman3d.core.visualization import visualize_smpl_hmr
    gender = 'female'
    focal_length = 5000
    det_width = 224
    det_height = 224

    # 可以传入smpl的poses参数、betas参数和gender参数
    body_model_config = dict(
        type='smpl', model_path=model_path, gender=gender)
    visualize_smpl_hmr(
        poses=poses,
        betas=betas,
        bbox=bbox,
        body_model_config=body_model_config,
        focal_length=focal_length,
        det_width=det_width,
        det_height=det_height,
        T=cam_translation,
        output_path='hmr_demo.mp4',
        origin_frames=image_folder,
        resolution=(1024, 1024))

    # 或者可以传入顶点
    body_model_config = dict(
        type='smpl', model_path=model_path, gender=gender)
    visualize_smpl_hmr(
        verts=verts,
        bbox=bbox,
        focal_length=focal_length,
        body_model_config=body_model_config,
        det_width=det_width,
        det_height=det_height,
        T=cam_translation,
        output_path='hmr_demo.mp4',
        origin_frames=image_folder,
        resolution=(1024, 1024))

    # 也可以传入二维关键点以替代bbox
    body_model_config = dict(
        type='smpl', model_path=model_path, gender=gender)
    visualize_smpl_hmr(
        verts=verts,
        body_model_config=body_model_config,
        kp2d=kp2d,
        focal_length=focal_length,
        det_width=det_width,
        det_height=det_height,
        T=cam_translation,
        output_path='hmr_demo.mp4',
        origin_frames=image_folder,
        resolution=(1024, 1024))
    ```

- **使用opencv相机可视化smpl模型:**
    传入`OpenCV`定义的内参矩阵`K`和外参矩阵`R`、`T`。
    ```python
    from mmhuman3d.core.visualization import visualize_smpl_calibration
    body_model_config = dict(
        type='smpl', model_path=model_path, gender=gender)
    visualize_smpl_calibration(
        poses=poses,
        betas=betas,
        transl=transl,
        body_model_config=body_model_config,
        K=K,
        R=R,
        T=T,
        output_path='opencv.mp4',
        origin_frames='bg_video.mp4',
        resolution=(1024, 1024))
    ```

### 不同的render_choice:
- **可视化mesh:**
    可以直接将 `render_choice` 设置为 `hq`(高质量), `mq`(中等质量) or `lq`(低质量)。

- **可视化二值轮廓:**
    可以直接将 `render_choice` 设置为 `silhouette`。 输出的视频和图片会是二值掩膜。

- **可视化身体部件的轮廓:**
    可以直接将 `render_choice` 设置为 `part_silhouette`。 输出的视频和图片会是身体部件的分割掩膜。

- **可视化深度图:**
    可以直接将 `render_choice` 设置为 `depth`。 输出的视频和图片会是灰色的深度图。

- **可视化法线贴图:**
    可以直接将 `render_choice` 设置为 `normal`。

- **可视化点云:**
    可以直接将 `render_choice` 设置为 `pointcloud`。 输出的视频和图片会是包含关键点的点云。

- **颜色选择:**
    指定调色板为 'white', 'black', 'blue', 'green', 'red', 'yellow', 并且传入一个长度为num_person的字符串列表。
    或者传入形状为(num_person, 3)的`numpy.ndarray`。该向量应该进行归一化， (1.0, 1.0, 1.0) 表示白色, 其色彩通道的顺序为RGB。

- **可微渲染:**
    指定 `no_grad=False` 和 `return_tensor=True`.

### 重要的参数:
- **背景图片:**
    传入 `image_array`(形状为(frame, h, w, 3)的`numpy.ndarray`) 或 `frame_list`(包含.png 或 .jpg图像的路径`list`) 或 `origin_frames`(视频路径或文件夹路径的字符串)。 顺序的优先级为 `image_array` > `frame_list` > `origin_frames`。
    如果背景图片过大，应该指定`read_frames_batch` 为 `True`，以减轻输入输出的负担。如果视频的帧数大于等于500，上述设置会在代码中自动完成。

- **smpl模型的pose和顶点:**
    可以采用以下两种方式传入smpl模型的mesh信息:
    1). 传入`poses`、 `betas`(可选)、 `transl`(可选) 和 `gender`(可选)。
    2). 也可以直接传入`verts`。如果采用传入`verts`的方式，由于需要获得`faces`，您需要指定`body_model` 或 `model_path`。
    优先级顺序为`verts` > (`poses` & `betas` & `transl` & `gender`)
    3). 对于多人的重建，您需要指定一个额外的维度`num_person`, 表示人的数量。例如, smpl模型的`verts` 的形状为 (num_frame, num_person, 6890, 3), smpl模型的`poses`参数的形状为 (num_frame, num_person, 72), smpl模型的`betas`参数的形状为 (num_frame, num_person, 10), vibe的`pred_cam` 的形状为 (num_frame, num_person, 3)。由于相机矩阵`K`, `R`和`T`是适用于每一帧的，上述操作并不会对它们造成影响。

- **身体模型:**
    有两种传入人体模型的方式:
    1). 传入包含与`build_body_model`相同配置的字典`body_model_config`。
    2). 直接传入`body_model`。优先级顺序为`body_model` > (`model_path` & `model_type` & `gender`)。

- **输出路径:**
    输出路径可以为视频文件(类型为`None` 或 `str`)或图像文件夹的路径(类型为`str`)。
    1). 如果输出路径指定为`None`, 不会写入输出文件。
    2). 如果输出路径指定为`xxx.mp4`, 将还会写入一个视频。请确保有足够的空间用于存放时序图像，这些图像会自动删除。
    3). 如果输出路径指定为图像文件夹`xxx/`, 将会创建一个文件夹，并将图像存放在文件夹里面。
