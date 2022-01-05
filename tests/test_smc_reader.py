from mmhuman3d.data.data_structures.smc_reader import SMCReader

TEST_SMC_PATH = 'tests/data/mocap/p000103_a000011_tiny.smc'


def test_get_calibration_dict():
    smc = SMCReader(TEST_SMC_PATH)

    calibration_dict = smc.calibration_dict
    assert '0' in calibration_dict.keys(
    ), 'Calibration should contain at least a kinect'


def test_get_num_frames():
    smc = SMCReader(TEST_SMC_PATH)

    num_kinect_frame = smc.get_kinect_num_frames()
    assert num_kinect_frame >= 1, 'SMC file contain at least one frame'

    num_iphone_frame = smc.get_iphone_num_frames()
    assert num_iphone_frame >= 1, 'SMC file contain at least one frame'


def test_get_kinect_extrinsics():
    smc = SMCReader(TEST_SMC_PATH)

    color_extrinsics = smc.get_kinect_color_extrinsics(0, homogeneous=True)
    assert color_extrinsics.shape == (
        4, 4), 'Kinect Color Extrinsic should be a matrix with shape 4x4'

    color_extrinsics = smc.get_kinect_color_extrinsics(0, homogeneous=False)
    assert color_extrinsics['R'].shape == (
        3, 3), 'Kinect Color R should be a matrix with shape 3x3'
    assert color_extrinsics['T'].shape == (
        3, ), 'Kinect Color T should be a matrix with shape 3'

    depth_extrinsics = smc.get_kinect_depth_extrinsics(0, homogeneous=True)
    assert depth_extrinsics.shape == (
        4, 4), 'Kinect Depth Extrinsic should be a matrix with shape 4x4'

    depth_extrinsics = smc.get_kinect_depth_extrinsics(0, homogeneous=False)
    assert depth_extrinsics['R'].shape == (
        3, 3), 'Kinect Depth R should be a matrix with shape 3x3'
    assert depth_extrinsics['T'].shape == (
        3, ), 'Kinect Depth T should be a matrix with shape 3'


def test_get_kinect_intrinsics():
    smc = SMCReader(TEST_SMC_PATH)

    color_intrinsics = smc.get_kinect_color_intrinsics(0)
    assert color_intrinsics.shape == (
        3, 3), 'Kinect Color Intrinsic should be a matrix with shape 3x3'

    depth_intrinsics = smc.get_kinect_depth_intrinsics(0)
    assert depth_intrinsics.shape == (
        3, 3), 'Kinect Depth Intrinsic should be a matrix with shape 3x3'


def test_get_kinect_resolution():
    smc = SMCReader(TEST_SMC_PATH)

    color_resolution = smc.get_kinect_color_resolution(0)
    assert color_resolution.shape == (
        2, ), 'Kinect Color Resolution should be a 2D matrix'

    depth_resolution = smc.get_kinect_depth_resolution(0)
    assert depth_resolution.shape == (
        2, ), 'Kinect Depth Resolution should be a 2D matrix'


def test_get_kinect_data():
    smc = SMCReader(TEST_SMC_PATH)

    kinect_color = smc.get_kinect_color(0, frame_id=0)
    assert kinect_color.shape == (
        1, 1080, 1920,
        3), 'Kinect Color should have resolution of 1x1080x1920x3'

    kinect_depth = smc.get_kinect_depth(0, frame_id=0)
    assert kinect_depth.shape == (
        1, 576, 640), 'Kinect depth should have resolution of 1x576x640'

    mapped_color, depth = smc.get_kinect_rgbd(0, 0)
    assert mapped_color.shape == (
        576, 640, 3), 'Kinect RGBD should have color resolution of 576x640x3'


def test_get_iphone_extrinsics():
    smc = SMCReader(TEST_SMC_PATH)

    iphone_extrinsics = smc.get_iphone_extrinsics(iphone_id=0, frame_id=0)
    assert iphone_extrinsics.shape == (
        4, 4), 'Iphone Color Extrinsic should be a matrix with shape 4x4'


def test_get_iphone_intrinsics():
    smc = SMCReader(TEST_SMC_PATH)

    iphone_intrinsics = smc.get_iphone_intrinsics(iphone_id=0, frame_id=0)
    assert iphone_intrinsics.shape == (
        3, 3), 'Iphone Color Intrinsic should be a matrix with shape 3x3'


def test_get_iphone_resolution():
    smc = SMCReader(TEST_SMC_PATH)

    color_resolution = smc.get_iphone_color_resolution(0)
    assert color_resolution.shape == (
        2, ), 'iPhone Color Resolution should be a 2D matrix'


def test_get_iphone_data():
    smc = SMCReader(TEST_SMC_PATH)

    iphone_color = smc.get_iphone_color(iphone_id=0, frame_id=0)
    assert iphone_color.shape == (
        1, 1440, 1920, 3), 'iPhone color image should have shape 1x1440x1920x3'

    iphone_depth = smc.get_iphone_depth(iphone_id=0, frame_id=0)
    assert iphone_depth.shape == (
        1, 192, 256), 'iPhone depth image should have shape 1x192x256'


def test_get_mask():
    smc = SMCReader(TEST_SMC_PATH)

    depth_mask = smc.get_depth_mask(0, 0)
    assert depth_mask.shape == (576,
                                640), 'depth mask should have shape 576x640'

    kinect_mask = smc.get_kinect_mask(0, 0)
    assert kinect_mask.shape == (576,
                                 640), 'Kinect mask should have shape 576x640'


def test_get_kinect_transformation():
    smc = SMCReader(TEST_SMC_PATH)

    trans_d2c = smc.get_kinect_transformation_depth_to_color(0)
    assert trans_d2c.shape == (
        4, 4), 'Depth to Color Transformation matrix should have shape 4x4'

    trans_c2d = smc.get_kinect_transformation_color_to_depth(0)
    assert trans_c2d.shape == (
        4, 4), 'Color to Depth Transformation matrix should have shape 4x4'


def test_get_kinect_skeleton_3d():
    smc = SMCReader(TEST_SMC_PATH)

    skeleton_3d = smc.get_kinect_skeleton_3d(0, 0)
    assert len(skeleton_3d) != 0, 'skeleton should contain some data'


def test_get_depth_floor():
    smc = SMCReader(TEST_SMC_PATH)

    depth_floor = smc.get_depth_floor(0)
    assert 'center' in depth_floor.keys(
    ), 'center should be present in depth floor parameters'
