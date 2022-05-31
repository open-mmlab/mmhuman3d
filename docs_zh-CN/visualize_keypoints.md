## 关键点可视化

### 2D关键点可视化
- **2D关键点可视化的简单示例:**

    假定拥有一个维度为(10, 133, 2)的`coco_wholdbody`关键点.
    ```python
    from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d

    visualize_kp2d(
        kp2d_coco_wholebody,
        data_source='coco_wholebody',
        output_path='some_video.mp4',
        resolution=(1024, 1024))
    ```
    'some_video.mp4'中会储存大小为1024x1024的10帧视频

- **data_source 和 mask:**

    如果关键点中有一些并没有什么意义，您应该提供mask。
    `data_source` 主要用来搜索肢体的连接关系和上色方式。如果您的数据集在[convention](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/)中，您应该指定`data_sourece`.
    例如, 将`coco_wholebody`的关键点转换为`smpl`，并将其可视化:
    ```python
    from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
    from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d

    kp2d_smpl, mask = convert_kps(kp2d_coco_wholebody, src='coco_wholebody', dst='smpl')
    visualize_kp2d(
        kp2d_smpl,
        mask=mask,
        output_path='some_video.mp4',
        resolution=(1024, 1024))
    ```
    `mask` 默认为 `None` 。当您确定所有关键点都有效时，请忽略它。


- **是否在背景上进行显示:**

    或许您想用numpy输入背景

    例如，您想将一个形状为(10, 133, 2)的`coco_wholebody`二维关键点可视化为`smpl`规范。
    ```python
    from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
    from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d

    background = np.random.randint(low=0, high=255, shape=(10, 1024, 1024, 4))
    # 多人, 其形状为 (num_person, num_joints, 2)
    out_image = visualize_kp2d(
        kp2d=kp2d, image_array=background, data_source='coco_wholebody', return_array=True)

    ```
    这只是一个例子，您可以灵活使用这个功能。

    如果您想要在每一帧上画出关键点，您需要提供`frame_list`(包含图片路径的列表)或者`origin_frames`(视频路径或存放图像的文件夹的路径)。**注意，`frame_list`中的顺序必须根据名称进行排序**
    ```python
    frame_list = ['im1.png', 'im2.png', ...]
    visualize_kp2d(
        kp2d_coco_wholebody,
        data_source='coco_wholebody',
        output_path='some_video.mp4',
        resolution=(1024, 1024),
        frame_list=frame_list)

    origin_frames = 'some_folder'
    visualize_kp2d(
        kp2d_coco_wholebody,
        data_source='coco_wholebody',
        output_path='some_video.mp4',
        resolution=(1024, 1024),
        origin_frames=origin_frames)

    origin_frames = 'some.mp4'
    array = visualize_kp2d(
        kp2d_coco_wholebody,
        data_source='coco_wholebody',
        output_path='some_video.mp4',
        resolution=(1024, 1024),
        return_array=True,
        origin_frames=origin_frames)

    ```

- **输出视频或图像:**

    如果 `output_path` 是文件夹, 该函数会输出图像。
    如果 `output_path` 是'.mp4'文件的路径, 该函数会输出视频。
    当 `return_array` 为 `True` 时，`output_path` 可以被设置为 `None` 。 该函数会输出一个形状为(frame, width, height, 3)的数组。

- **是否在图像中显示原始帧的名称:**

    通过指定 `with_file_name``=True`, 图像中会显示原始帧的名称。

- **规范中不支持的数据集或者可视化某些特定的肢体**

    如果[规范](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/)中不支持您的数据集，您应该按照如下的格式提供肢体的顺序:
    `limbs=[[0, 1], ..., [10, 11]]`

- **其他的参数:**

    请阅读源码中的注释，这很容易理解。

### 3D关键点可视化

- **单人可视化的简单例子:**

    假定拥有一个形状为(num_frame, 144, 3)的`smplx`规范的三维关节点。
    ```python
    visualize_kp3d(kp3d=kp3d, data_source='smplx', output_path='some_video.mp4')
    ```
    输出的视频中会有一个可视化的人体，其每个身体部位用不同的颜色加以区分。

- **多人可视化的简单例子:**

    假定拥有两个`smplx`规范的三维关节点，其形状皆为(num_frame, 144, 3)。
    ```python
    kp3d = np.concatenate([kp3d_1[:, np.newaxis], kp3d_2[:, np.newaxis]], axis=1)
    # kp3d的形状现在为(num_frame, num_person, 144, 3)
    visualize_kp3d(kp3d=kp3d, data_source='smplx', output_path='some_video.mp4')
    ```
    输出的视频中会有两个人可视化的人体，每个人都是纯色的，并且会有一个颜色图例描述每个人的索引。

- **data_source 和 mask:**

    与[visualize_kp2d](#visualize_kp2d)相同。

- **规范中不支持的数据集或者可视化某些特定的肢体:**

    与[visualize_kp2d](#visualize_kp2d)相同。

- **输出:**
    如果 `output_path` 是文件夹, 该函数会输出图像。
    如果 `output_path` 是'.mp4'文件的路径, 该函数会输出视频。
    当 `return_array` 为 `True` 时，`output_path` 可以被设置为 `None` 。 该函数会输出一个形状为(frame, width, height, 3)的数组。

- **其他参数:**

    请阅读源码中的注释，这很容易理解。


### ffmpeg_utils
- 在[ffmpeg_utils](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/utils/ffmpeg_utils.py)中, 每个函数有充分的描述, 并且函数名称根据用途定义, 很容易理解。

- **读文件:**

    images_to_array, video_to_array

- **写文件:**

    array_to_images, array_to_video

- **格式转换:**

    gif_to_images, gif_to_video,  video_to_images, video_to_gif, images_to_gif, images_to_video

- **时序裁剪和拼接:**

    slice_video, temporal_concat_video

- **空间裁剪和拼接:**

    crop_video, spatial_concat_video

- **压缩:**

    compress_gif, compress_video
