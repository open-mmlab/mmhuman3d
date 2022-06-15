## Visualize SMPL Mesh
- **fast visualize smpl(x) pose without background images:**

    You have smpl pose tensor or array shape of which is (frame, 72)
    ```python
    from mmhuman3d.core.visualization import visualize_smpl_pose
    body_model_config = dict(
        type='smpl', model_path=model_path)
    visualize_smpl_pose(
        poses=poses,
        output_path='smpl.mp4',
        resolution=(1024, 1024))
    ```

    Or you have smplx pose tensor or array shape of which is (frame, 165)
    ```python
    body_model_config = dict(
        type='smplx', model_path=model_path)
    visualize_smpl_pose(
        poses=poses,
        body_model_config=body_model_config,
        output_path='smplx.mp4',
        resolution=(1024, 1024))
    ```
    You could also feed dict tensor of smplx definitions. You could check that in [visualize_smpl](mmhuman3d/core/visualization/visualize_smpl.py#L166-211) or [original smplx](https://github.com/vchoutas/smplx/blob/master/smplx/body_models.py).


- **visualize T-pose:**
    If you want to visualize a T-pose smpl or your poses do not have global_orient, you can do:
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

- **visualize smpl with predicted VIBE camera:**
    You have poses (numpy/tensor) of shape (frame, 72), betas of shape (frame, 10), pred_cam of shape (10, 4).
    E.g., we use vibe sample_video.mp4 as an example.
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

    # pass pred_cam & bbox
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

    # or pass orig_cam
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

- **visualize smpl with predicted HMR/SPIN camera:**
    You have poses (numpy/tensor) of shape (frame, 72), betas of shape (frame, 10), cam_translation of shape (10, 4).
    E.g., we use vibe sample_video.mp4 as an example.
    ```python
    import pickle
    from mmhuman3d.core.visualization import visualize_smpl_hmr
    gender = 'female'
    focal_length = 5000
    det_width = 224
    det_height = 224

    # you can pass smpl poses & betas & gender
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

    # or you can pass verts
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

    # you can also pass kp2d in replace of bbox.
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

- **visualize smpl with opencv camera:**
    You should pass the opencv defined intrinsic matrix K and extrinsic matrix R, T.
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

### Different render_choice:
- **visualize mesh:**
    This is independent of cameras and you could directly set `render_choice` as `hq`(high quality), `mq`(medium quality) or `lq`(low quality).

- **visualize binary silhouettes:**
    This is independent of cameras and you could directly set `render_choice` as `silhouette`. The output video/images will be binary masks.

- **visualize body part silhouette:**
    This is independent of cameras and you could directly set `render_choice` as `part_silhouette`. The output video/images will be body part segmentation masks.

- **visualize depth map:**
    This is independent of cameras and you could directly set `render_choice` as `depth`.
    The output video/images will be gray depth maps.

- **visualize normal map:**
    This is independent of cameras and you could directly set `render_choice` as `normal`.
    The output video/images will be colorful normal maps.

- **visualize point clouds:**
    This is independent of cameras and you could directly set `render_choice` as `pointcloud`.
    The output video/images will be point clouds with keypoints.

- **Choose your color:**
    Set palette as 'white', 'black', 'blue', 'green', 'red', 'yellow', and pass a list of string with the length of num_person.
    Or send a numpy.ndarray of shape (num_person, 3). Should be normalized color: (1.0, 1.0, 1.0) represents white. The color channel is RGB.

- **Differentiable render:**
    Set `no_grad=False` and `return_tensor=True`.

### Important parameters:
- **background images:**
    You could pass `image_array`(`numpy.ndarray` of shape (frame, h, w, 3)) or `frame_list`(`list` of paths of images(.png or .jpg)) or `origin_frames`(str of video path or image folder path). The priority order is `image_array` > `frame_list` > `origin_frames`.
    If the background images are too big, you should set `read_frames_batch` as `True` to relieve the IO burden. This will be done automatically in the code when you number of frame is large than 500.

- **smpl pose & verts:**
    There area two ways to pass smpl mesh information:
    1). You pass `poses`, `betas`(optional) and `transl`(optional) and `gender`(optional).
    2). You pass `verts` directly and the above three will be ignored. The `body_model` or `model_path` is still required if you pass`verts` since we need to get the `faces`.
    The priority order is `verts` > (`poses` & `betas` & `transl` & `gender`).
    Check the docstring for details.
    3). for multi-person, you should have an extra dim for num_person. E.g., shape of smpl `verts` should be (num_frame, num_person, 6890, 3), shape of smpl `poses` should be (num_frame, num_person, 72), shape of smpl `betas` should be (num_frame, num_person, 10), shape of vibe `pred_cam` should be (num_frame, num_person, 3). This doesn't have influence on `K`, `R`, `T` since they are for every frame.

- **body model:**
    There are two ways to pass body model:
    1). You pass a dict `body_model_config` which containing the same configs as build_body_model
    2). You pass `body_model` directly and the above three will be ignored.
    The priority order is `body_model` > (`model_path` & `model_type` & `gender`).
    Check the docstring for details.

- **output path:**
    Output_path could be `None` or `str` of video path or `str` of image folder path.
    1). If `None`, no output file will be wrote.
    2). If a video path like `xxx.mp4`, a video file will be wrote. Make sure you have enough space for temporal images. The images will be removed automatically.
    3). If a image folder path like `xxx/`, a folder will be created and the images will be wrote into it.
