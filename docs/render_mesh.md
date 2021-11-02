### Visualize smpl mesh
- **fast visualize smpl(x) pose:**

    You have smpl pose tensor or array which shape is (frame, 72)
    ```python
    from mmhuman3d.core.visualization import visualize_smpl_pose
    visualize_smpl_pose(poses=poses, body_model_dir=body_model_dir, output_path='some_video.mp4', model_type='smpl', resolution=(1024, 1024))
    ```

    Or you have smplx pose tensor or array which shape is (frame, 165)
    ```python
    visualize_smpl_pose(poses=poses, body_model_dir=body_model_dir, output_path='some_video.mp4', model_type='smplx', resolution=(1024, 1024))
    ```
    You could also feed dict tensor of smplx definitions. You could check that in [visualize_smpl](mmhuman3d/core/visualization/visualize_smpl.py#L166-211) or [original smplx](https://github.com/vchoutas/smplx/blob/master/smplx/body_models.py).


- **visualize T-pose:**
    If you want to visualize a T-pose smpl or your poses do not have global_orient, you can do:
    ```python
    import torch
    from mmhuman3d.core.visualization import visualize_T_pose
    poses = torch.zeros(10, 72)
    visualize_T_pose(poses=poses, model_type='smpl', body_model_dir=body_model_dir, origin_frames='sample_video.mp4', orbit_speed=(1, 0.5))
    ```

- **visualize smpl with predicted camera:**
    You have poses (numpy/tensor) of shape (frame, 72), betas of shape (frame, 10), transl of shape (frame, 3), pred_cam of shape (10, 4).
    E.g., we use vibe sample_video.mp4 as an example.
    ```python
    import pickle
    from mmhuman3d.core.visualization import visualize_smpl_pred
    with open('vibe_output.pkl', 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    poses = d[1]['pose']
    pred_cam = d[1]['orig_cam']
    gender = 'female'
    visualize_smpl_pred(poses=poses, betas=betas, gender=gender, pred_cam=pred_cam, model_type='smpl', body_model_dir=body_model_dir, origin_frames='sample_video.mp4')
    ```

- **visualize smpl with opencv camera:**
    You should pass the opencv defined intrinsic matrix K and extrinsic matrix R, T.
    ```python
    from mmhuman3d.core.visualization import visualize_smpl_opencv
    visualize_smpl_opencv(poses=poses, betas=betas, gender=gender, model_type='smpl', K=K, R=R, T=T, body_model_dir=body_model_dir, origin_frames='sample_video.mp4')
    ```

### Different render_choice:
- **visualize mesh:**
    This is independent of cameras and you could directly set `render_choice` as `hq`(high quality), `mq`(medium quality) or `lq`(low quality).

- **visualize binary silhouettes:**
    This is independent of cameras and you could directly set `render_choice` as `silhouette`. The output video/images will be binary masks.

- **visualize body part silhouette:**
    This is independent of cameras and you could directly set `render_choice` as `part_silhouette`. The output video/images will be body part segementation masks.

- **visualize depth map:**
    This is independent of cameras and you could directly set `render_choice` as `depth`.
    The output video/images will be gray depth maps.

- **visualize point clouds:**
    This is independent of cameras and you could directly set `render_choice` as `pointcloud`.
    The output video/images will be point clouds with keypoints.

### Differentiable render

-**Differentiable render with different choice:**
    You can set `render_choice` as former, this requires larger GPU memory and the render choice will be returned as a requires_grad `Tensor` of shape (frames, h, w, n_class) for `part_silhouette`, of shape (frames, h, w) for `silhouette`, and of shape (frames, h, w, 4) for others.
    `neural_render_smpl` could recieve the same parameters as the above but will return a `tensor`.

### Important parameters:
-**background images:**
    You could pass `image_array`(`numpy.ndarray` of shape (frame, h, w, 3)) or `frame_list`(`list` of paths of images(.png or .jpg)) or `origin_frames`(str of video path or image folder path). The priority order is `image_array` > `frame_list` > `origin_frames`.
    If the background images are too big, you should set `read_frames_batch` as `True` to relieve the IO burden. This will be done automatically in the code when you number of frame is large than 500.

-**smpl pose & verts:**
    There area two ways to pass smpl mesh information:
    1). You pass `poses`, `betas`(optional) and `transl`(optional) and `gender`(optional).
    2). You pass `verts` directly and the above three will be ignored. The `body_model` or `body_model_dir` is still required if you pass`verts` since we need to get the `faces`.
    The priority order is `verts` > (`poses` & `betas` & `transl` & `gender`).
    Check the docstring for details.


-**body model:**
    There area two ways to pass body model:
    1). You pass `body_model_dir`, `model_type`(optional) and `gender`(optional).
    2). You pass `body_model` directly and the above three will be ignored.
    The priority order is `body_model` > (`body_model_dir` & `model_type` & `gender`).
    Check the docstring for details.

-**output path:**
    Output_path could be `None` or `str` of video path or `str` of image folder path.
    1). If `None`, no output file will be wrote.
    1). If a video path like `xxx.mp4`, a video file will be wrote. Make sure you have enough space for temporal images. The images will be removed automatically.
    1). If a image folder path like `xxx/`, a folder will be created and the images will be wrote into it.
