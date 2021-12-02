## Visualize Keypoints

### Visualize 2d keypoints
- **simple example for visualize 2d keypoints:**

    You have 2d coco_wholebody keypoints of shape(10, 133, 2).
    ```python
    from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d

    visualize_kp2d(
        kp2d_coco_wholebody,
        data_source='coco_wholebody',
        output_path='some_video.mp4',
        resolution=(1024, 1024))
    ```
    Then a 1024x1024 sized video with 10 frames would be save as 'some_video.mp4'

- **data_source and mask:**

    If your keypoints have some nonsense points, you should provide the mask. `data_source` is mainly used to search the limb connections and palettes. You should specify the data_source if you dataset is in [convention](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/).
    E.g., convert coco_wholebody keypoints to the convention of smpl and visualize it:
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
    mask is `None` by default. This is the same as all ones mask, then no keypoints will be excluded. Ignore it when you are sure that all the keypoints are valid.


- **whether plot on backgrounds:**

    Maybe you want to use numpy input backgrounds.

    E.g., you want to visualize you coco_wholebody kp2d as smpl convention. You have 2d coco_wholebody keypoints of shape(10, 133, 2).
    ```python
    from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
    from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d

    background = np.random.randint(low=0, high=255, shape=(10, 1024, 1024, 4))
    # multi_person, shape is (num_person, num_joints, 2)
    out_image = visualize_kp2d(
        kp2d=kp2d, image_array=background, data_source='coco_wholebody', return_array=True)

    ```
    This is just an example, you can use this function flexibly.

    If want to plot keypoints on frame files, you could provide `frame_list`(list of image path). **Be aware that the order of the frame will be sorted by name.**
    or `origin_frames`(mp4 path or image folder path), **Be aware that you should provide the correct `img_format` for `ffmpeg` to read the images.**.
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
    The superiorty of background images: `frame_list`

- **output a video or frames:**

    If `output_path` is a folder, this function will output frames.
    If `output_path` is a '.mp4' path, this function will output a video.
    `output_path` could be set as `None` when `return_array` is True. The function will return an array of shape (frame, width, height, 3).

- **whether plot origin file name on images:**

    Specify `with_file_name=True` then origin frame name will be plotted on the image.

- **dataset not in existing convention or want to visualize some specific limbs:**

    You should provide limbs like
    `limbs=[[0, 1], ..., [10, 11]]`
    if you dataset is not in [convention](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/core/conventions/keypoints_mapping/).

- **other parameters:**

    Easy to understand, please read the doc strings in the function.

### Visualize 3d keypoints

- **simple example for visualize single person:**

    You have kp3d in smplx convention of shape (num_frame, 144, 3).
    ```python
    visualize_kp3d(kp3d=kp3d, data_source='smplx', output_path='some_video.mp4')
    ```
    The result video would have one person dancing, each body part has its own color.

- **simple example for visualize multi person:**

    You have kp3d_1 and kp3d_2 which are both in smplx convention of shape (num_frame, 144, 3).
    ```python
    kp3d = np.concatenate([kp3d_1[:, np.newaxis], kp3d_2[:, np.newaxis]], axis=1)
    # kp3d.shape is now (num_frame, num_person, 144, 3)
    visualize_kp3d(kp3d=kp3d, data_source='smplx', output_path='some_video.mp4')
    ```
    The result video would have two person dancing, each in a pure color, and the there will be a color legend describing the index of each person.

- **data_source and mask:**

    The same as [visualize_kp2d](#visualize_kp2d)

- **dataset not in existing convention or want to visualize some specific limbs:**

    The same as [visualize_kp2d](#visualize_kp2d)

- **output:**
    If `output_path` is a folder, this function will output frames.
    If `output_path` is a '.mp4' path, this function will output a video.
    `output_path` could be set as `None` when `return_array` is True. The function will return an array of shape (frame, width, height, 3).

- **other parameters:**

    Easy to understand, please read the doc strings in the function.


### About ffmpeg_utils
- In [ffmpeg_utils](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/utils/ffmpeg_utils.py) , each function has abundant doc strings, and the semantically defined function names could be easily understood.

- **read files:**

    images_to_array, video_to_array

- **write files:**

    array_to_images, array_to_video

- **convert formats:**

    gif_to_images, gif_to_video,  video_to_images, video_to_gif, images_to_gif, images_to_video

- **temporally crop/concat:**

    slice_video, temporal_concat_video

- **spatially crop/concat:**

    crop_video, spatial_concat_video

- **compress:**

    compress_gif, compress_video
