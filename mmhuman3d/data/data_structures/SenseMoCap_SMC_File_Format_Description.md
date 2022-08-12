# SMC(SenseMoCap) File Format Description

SMC (SenseMoCap) is a file format designed with multi-camera multi-model support in mind. Each smc file is essentially a HDF5 database，made easy for cross-platform, cross-language support (h5py, H5Cpp).

Each SMC file contains the following structure.

- ### *.smc (File)

  - Attributes

    - actor_id: actor id, int32 scalar
    - action_id: action id, int32 scalar
    - datetime_str: data collection time stamp, string (YYYY-MM-DD-hh-mm-ss)

  - ### Extrinsics (Dataset):

    - A JSON String with N calibrated Kinects and M iPhone extrinsic parameters with
      - Indexing cameras:
        - Kinect Color Index: i*2,  0<=i<N
        - Kinect Depth Index: i*2 + 1,  0<=i<N
        - iPhone Index: N*2 + j,  0<=j<M
      - Parameters:
        - Extrinsic: cam2world transformation
          - R: Rotation Matrix [3,3]
          - T: Translation [3]
          - Floor: floor parameter [4]

  - ### Kinect (Group)

    - Attributes

      - num_frame: Kinect: uint32 scalar
      - num_device: Kinect: uint8 scalar
      - depth_mode: Kinect depth mode, uint8 scalar, as specified in K4A SDK [enum](https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___enumerations_ga3507ee60c1ffe1909096e2080dd2a05d.html)
      - color_resolution: Kinect RGB color mode uint8 scalar, as specified in K4A SDK[enum](https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___enumerations_gabc7cab5e5396130f97b8ab392443c7b8.html#gabc7cab5e5396130f97b8ab392443c7b8)

    - #### KinectID:  (HDF5 Group), range from 0 to N, each stores data collect from Kinect camera

      - ##### Calibration (HDF5 Group)

        - ###### Color (HDF5 Group)

          - Intrinsics (Dataset): K4A SDK factory calibrated intrinsic, float32, shape (15,)
          - Resolution (Dataset): color camera resolution (width, height), uint16, shape (2,),
          - MetricRadius (Dataset): metric radius from K4A SDK, float32 scalar

        - ###### Depth (HDF5 Group)

          - Intrinsics (Dataset): K4A SDK factory calibrated intrinsic, float32, shape (15,)
          - Resolution (Dataset): color camera resolution (width, height), uint16, shape (2,),
          - MetricRadius (Dataset): metric radius from K4A SDK, float32 scalar

      - ##### Color (Group)

        - Dataset with F(number of frames) color images
          - RGBA Color image (byte array)

      - ##### Depth (Group)

        - Dataset with F(number of frames) depth images
          - 16 bit depth image:  2D uint16 array with Shape H*W (576, 640)

      - ##### IR (Group)

        - Dataset with F(number of frames) Infrared images
          - 16 bit IR image:  2D uint16 array with Shape H*W (576, 640)

      - ##### Mask (Group)

        - Dataset with F(number of frames) body mask images from frame difference
          - 8 bit body mask image:  2D uint8 array with Shape H*W (576, 640)

      - ##### Mask_k4abt (Group)

        - Dataset with F(number of frames) body mask images from K4A Body Tracking SDK
          - 8 bit body mask image:  2D uint8 array with Shape H*W (576, 640)

      - ##### Skeleton_k4abt (Group)

        - Dataset with F(number of frames) Skeleton data from K4A Body Tracking SDK
          - JSON: as specified in K4A SDK

      - ##### Background (Group)

        - Background Images for matting. Available from v3 data
          - Color : Same as Kinect Color
          - Depth: Same as Kinect Depth

  - ### iPhone (Group)

    - Attributes

      - num_frame: number of iPhone frames, int32 scalar, close to number of Kinect frames * 2 + 4
      - color_resolution: iPhone RGB resolution(width, height), int32, shape (2,)
      - depth_resolution: iPhone Depth resolution (width, height), int32, shape (2,)

    - #### iPhoneID (Group)

      - ##### Color (Group)

        - Dataset with F(number of frames) color images
          - RGBA Color image (byte array)

      - ##### Depth (Group)

        - Dataset with F(number of frames) depth images (from iPhone LiDAR)
          - 16 bit depth image:  2D uint16 array with Shape H*W (192, 256)

      - ##### Confidence (Group)

        - Dataset with F(number of frames) confidence maps
          - 8 bit confidence:  2D uint16 array with Shape H*W (192, 256)

      - ##### Mask_ARKit (Group)

        - Dataset with F(number of frames) body mask from Apple ARKit
          - 8 bit body mask:  2D uint16 array with Shape H*W (192, 256)

      - ##### CameraInfo (Group)

        - Dataset with F(number of frames) camera information
          - JSON with camera intrinsics, timestamp etc
          -

  - ### Keypoints3D (Group)

    - Attributes
      - num_frame: number of frames for 3D key points
      - convention: convention for key points
      - created_time: creation timestamp
    - keypoints3d (Dataset): 3D key point computed from triangulate_optim
    - Keypoints3d_mask (Dataset): corresponding mask

  ### Keypoints2D

  - Kinect (Group)

    - #### DeviceID (Group)

      - DeviceID
        - Length aligned with 3D Keypoints，reprojection from 3D Keypoints

  - iPhone (Group)

    - #### DeviceID (Group)

      - DeviceID
        - Length aligned with 3D Keypoints，reprojection from 3D Keypoints

  - ### SMPL (Group)

    - Attributes
      - num_frame: SMPL frames
      - created_time: creation timestamp
    - global_orient (Dataset): Global Orientation: Nx3
    - body_pose (Dataset): Body Pose: Nx23x3
    - betas (Dataset): SMPL Betas: 1x10
    - transl (Dataset): Global Translation:  Nx3
    - keypoints3d (Dataset): SMPL Keypoints: Nx3
