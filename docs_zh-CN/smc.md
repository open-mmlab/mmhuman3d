# SMC(SenseMoCap) 文件格式描述

SMC (SenseMoCap) 是一个设计用来存储多目多模态的文件格式。每一个SMC本质上是一个HDF5的数据库，支持不同平台，不同编程语言的读写。

每个SMC的机构约定如下:

- ### *.smc (File)

  - Attributes

    - actor_id: 演员编号, int32 字符
    - action_id: 动作编号, int32 字符
    - datetime_str: 数据采集时间戳, 字符串 (YYYY-MM-DD-hh-mm-ss)

  - ### Extrinsics (Dataset):

    - 一个存储N个Kinect和M个iPhone标定数据的JSON字符串
      - Indexing cameras:
        - Kinect RGB相机编号是偶数: i*2,  0<=i<N
        - Kinect 深度相机编号是奇数: i*2 + 1,  0<=i<N
        - iPhone 相机编号在Kinect后递增: N*2 + j,  0<=j<M
      - 参数:
        - 相机外参: 相机坐标系到世界坐标系的转换矩阵(cam2world)
          - R: 旋转矩阵 [3,3]
          - T: 平移向量 [3]
          - Floor: 地面参数 [4]

  - ### Kinect (Group)

    - Attributes

      - num_frame: 帧数: uint32 scalar
      - num_device: 场景中Kinect相机个数: uint8 scalar
      - depth_mode: Kinect相机深度模式, uint8 scalar, 来自 K4A SDK [enum](https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___enumerations_ga3507ee60c1ffe1909096e2080dd2a05d.html)
      - color_resolution: Kinect 彩色相机模式, uint8 scalar, 来自 K4A SDK[enum](https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___enumerations_gabc7cab5e5396130f97b8ab392443c7b8.html#gabc7cab5e5396130f97b8ab392443c7b8)

    - #### KinectID:  (HDF5 Group), 从 0 到 N, 每一个存储来自一个Kinect的所有数据

      - ##### Calibration (HDF5 Group)

        - ###### Color (HDF5 Group)

          - Intrinsics (Dataset): 使用K4A SDK读取的出厂内参标定矩阵, float32, shape (15,)
          - Resolution (Dataset): 彩色相机的分辨率(宽, 高), uint16, shape (2,),
          - MetricRadius (Dataset): K4A SDK中读取的Metric Radius, float32 scalar

        - ###### Depth (HDF5 Group)

          - Intrinsics (Dataset): 使用K4A SDK读取的出厂外参标定矩阵, float32, shape (15,)
          - Resolution (Dataset): 彩色相机的分辨率(宽, 高), uint16, shape (2,),
          - MetricRadius (Dataset): K4A SDK中读取的Metric Radius, float32 scalar

      - ##### Color (Group)

        - 存有F帧RGB图像的Dataset
          - RGBA 彩色图像 (byte array)

      - ##### Depth (Group)

        - 存有F帧深度图像的Dataset
          - 16位深度图:  2D uint16 array with Shape H*W (576, 640)

      - ##### IR (Group)

        - 存有F帧红外图像的Dataset
          - 16位红外图:  2D uint16 array with Shape H*W (576, 640)

      - ##### Mask (Group)

        - 存有F帧掩膜图像的Dataset(通过计算当前帧与背景帧的差别得出)
          - 8 bit body mask image:  2D uint8 array with Shape H*W (576, 640)

      - ##### Mask_k4abt (Group)

        - 存有F帧掩膜图像的Dataset(通过计使用K4A Body Tracking SDK得出)
          - 8 bit body mask image:  2D uint8 array with Shape H*W (576, 640)

      - ##### Skeleton_k4abt (Group)

        - 存有F帧骨骼关键点的Dataset(通过计使用K4A Body Tracking SDK得出)
          - JSON: as specified in K4A SDK

      - ##### Background (Group)

        - 背景图(v3及以上版本)
          - Color : 同 Kinect Color
          - Depth: 同 Kinect Depth

  - ### iPhone (Group)

    - Attributes

      - num_frame: iPhone帧数, int32 scalar, 大致等于Kinect帧数 * 2 + 4
      - color_resolution: iPhone 彩色相机的分辨率(宽, 高), int32, shape (2,)
      - depth_resolution: iPhone 深度相机的分辨率 (宽, 高), int32, shape (2,)

    - #### iPhoneID (Group)

      - ##### Color (Group)

        - 存有F帧RGB图像的Dataset
          - RGBA 彩色图像 (byte array)

      - ##### Depth (Group)

        - 存有F帧深度图像的Dataset (采集自iPhone LiDAR)
          - 16位深度图:  2D uint16 array with Shape H*W (192, 256)

      - ##### Confidence (Group)

        - 存有F帧ConfidenceMap的Dataset (采集自Apple ARKit)
          - 8 位 confidence:  2D uint16 array with Shape H*W (192, 256)

      - ##### Mask_ARKit (Group)

        - 存有F帧掩膜的Dataset (采集自Apple ARKit)
          - 8 bit body mask:  2D uint16 array with Shape H*W (192, 256)

      - ##### CameraInfo (Group)

        - 存有F帧相机参数的Dataset (采集自Apple ARKit)
          - JSON with camera intrinsics, timestamp etc
          - iPhone相机的内参每帧都会变, 需要每帧都存

  - ### Keypoints3D (Group)

    - Attributes
      - num_frame: 帧数
      - convention: 3D关键点格式
      - created_time: 创建时间戳
    - keypoints3d (Dataset): 使用triangulate_optim计算出来的3D关键点
    - Keypoints3d_mask (Dataset): 3D关键点掩膜

  ### Keypoints2D

  - Kinect (Group)

    - #### DeviceID (Group)

      - DeviceID
        - 从重投影算出的2D关键点

  - iPhone (Group)

    - #### DeviceID (Group)

      - DeviceID
        - 从重投影算出的2D关键点

  - ### SMPL (Group)

    - Attributes
      - num_frame: SMPL 帧数
      - created_time: 创建时间戳
    - global_orient (Dataset): Global Orientation: Nx3
    - body_pose (Dataset): Body Pose: Nx23x3
    - betas (Dataset): SMPL Betas: 1x10
    - transl (Dataset): Global Translation:  Nx3
    - keypoints3d (Dataset): SMPL Keypoints: Nx3
