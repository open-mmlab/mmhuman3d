## Visualization (version 0.2.0)

### Visualize 2d keypoints  
- **simple example for visualize 2d keypoints:**

    You have 2d mmpose keypoints of shape(10, 133, 2).
    ```python
    from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d
    visualize_kp2d(kp2d_mmpose, data_source='mmpose', output_path='some_video.mp4', resolution=(1024, 1024))
    ```
    Then a 1024x1024 sized video with 10 frames would be save as 'some_video.mp4'

- **data_source and mask:**

    If your keypoints have some nonsense points, you should provide the mask. `data_source` is mainly used to search the limb connections and palettes. You should specify the data_source if you dataset is in [convention](mmhuman3d/core/conventions/keypoints_mapping/).
    E.g., convert mmpose keypoints to the convention of smpl and visualize it:
    ```python
    from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
    from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d

    kp2d_smpl, mask = convert_kps(kp2d_mmpose, src='mmpose', dst='smpl')
    visualize_kp2d(kp2d_smpl, mask=mask, output_path='some_video.mp4', resolution=(1024, 1024))
    ```
    mask is `None` by default. This is the same as all ones mask, then no keypoints will be excluded. Ignore it when you are sure that all the keypoints are valid.

- **plot on a numpy array image:**

    It is easy to plot single frame.
    Or maybe you want to use numpy input frames.

    E.g., you want to visualize you mmpose kp2d as smpl convention.
    ```python
    from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
    from mmhuman3d.core.visualization.visualize_keypoints2d import plot_kp2d_frame
    from mmhuman3d.utils.keypoint_utils import search_limbs

    kp2d_smpl, mask = convert_kps(kp2d_mmpose, src='mmpose', dst='smpl')
    limbs, palette = search_limbs(data_source='smpl', mask=mask)

    canvas = np.zeros((1024, 1024, 3))

    # multi_person, shape is (num_person, num_joints, 2)
    if kp2d_smpl.ndim == 3:
        num_person, num_joints, _ = kp2d_smpl.shape
        for i in range(num_person):
            kp2d_person = kp2d_smpl[i]
            canvas = plot_kp2d_frame(kp2d_person=kp2d_person, canvas=canvas, limbs=limbs, palette=palette)

    # singe_person, shape is (num_joints, 2)
    elif kp2d_smpl.ndim == 2:
        canvas = plot_kp2d_frame(kp2d_person=kp2d_smpl,canvas=canvas, limbs=limbs, palette=palette)
    ```
    This is just an example, you can use this function flexibly.

- **whether plot on original frames:**

    If want to plot keypoints on original frames, you should provide `frame_list`. **Be ware that the order of the frame should correspond with your keypoints.**

    If you have an original mp4 video, you can slice the images then visualize the images with the `frame_list`:
    ```python
    import glob
    from mmhuman3d.utils.ffmpeg_utils import video_to_images

    video_to_images(input_path='some_video.mp4', output_folder='some_new_folder', img_format='%06d.png')

    frame_list = glob.glob('some_new_folder/*.png')
    frame_list.sort()

    visualize_kp2d(kp2d_mmpose, data_source='mmpose', output_path='some_video.mp4', resolution=(1024, 1024),
    frame_list=frame_list)
    ```

    `frame_list` is `None` by default. In this circumstance, the output video would self-ajust by the margin of keypoint coordinates `if resolution is None`.

- **output a video or frames:**

    If `output_path` is a folder, this function will output frames.
    If `output_path` is a '.mp4' path, this function will output a video.


- **whether plot origin file name on images:**

    Specify `with_file_name=True` then origin frame name will be plotted on the image.

- **dataset not in existing convention or want to visualize some specific limbs:**

    You should provide limbs like
    `limbs=[[0, 1], ..., [10, 11]]`
    if you dataset is not in [convention](mmhuman3d/core/conventions/keypoints_mapping/).

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

- **other parameters:**

    Easy to understand, please read the doc strings in the function.

### Visualize smpl mesh
- **fast visualize smpl(x) pose:**

    You have smpl pose tensor or array which shape is (frame, 72)
    ```python
    from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
    visualize_smpl_pose(poses=poses, body_model_dir=body_model_dir, output_path='some_video.mp4', model_type='smpl', render_choice='hq', resolution=(1024, 1024))
    ```

    Or you have smplx pose tensor or array which shape is (frame, 165)
    ```python
    visualize_smpl_pose(poses=poses, body_model_dir=body_model_dir, output_path='some_video.mp4', model_type='smplx', render_choice='hq', resolution=(1024, 1024))
    ```
    You could also feed dict tensor of smplx definitions. You could check that in [visualize_smpl](mmhuman3d/core/visualization/visualize_smpl.py#L166-211) or [original smplx](https://github.com/vchoutas/smplx/blob/master/smplx/body_models.py).


- **visualize T-pose:**

    ```TODO```

- **visualize smpl with predicted camera:**


- **visualize smpl with opencv camera:**

    ```TODO```

- **visualize binary silhouettes:**

    ```TODO```

- **visualize body part silhouette:**

    ```TODO```

### About ffmpeg_utils
- In [ffmpeg_utils](mmhuman3d/utils/ffmpeg_utils.py) , each function has abundant doc strings, and the semantically defined function names could be easily understood.

- **read files:**

    images_to_array, video_to_array

- **write files:**

    array_to_images, array_to_video

- **convert formats:**

    gif_to_images, gif_to_video,  video_to_images, video_to_gif, images_to_gif, images_to_video

- **temporally crop/concat:**

    temporal_crop_video, temporal_concat_video

- **spatially crop/concat:**

    spatial_crop_video, spatial_concat_video

- **compress:**

    compress_gif, compress_video

### Cameras(for v0.3.0)
- **WeakPerspectiveCameras:**

    ```TODO```
- **Convert cameras:**

    ```TODO```
