import numpy as np
import pytest

from mmhuman3d.data.data_structures.smc_reader import SMCReader

TEST_SMC_PATH = 'tests/data/dataset_sample/humman/p000003_a000014_tiny.smc'


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
    assert tuple(color_resolution) == (1920, 1080)

    depth_resolution = smc.get_kinect_depth_resolution(0)
    assert depth_resolution.shape == (
        2, ), 'Kinect Depth Resolution should be a 2D matrix'
    assert tuple(depth_resolution) == (640, 576)


def test_get_iphone_extrinsics():
    smc = SMCReader(TEST_SMC_PATH)

    iphone_extrinsics = smc.get_iphone_extrinsics(iphone_id=0)
    assert iphone_extrinsics.shape == (
        4, 4), 'Iphone Color Extrinsic should be a matrix with shape 4x4'


def test_get_iphone_intrinsics():
    smc = SMCReader(TEST_SMC_PATH)

    iphone_intrinsics = smc.get_iphone_intrinsics(iphone_id=0, frame_id=0)
    assert iphone_intrinsics.shape == (
        3, 3), 'Iphone Color Intrinsic should be a matrix with shape 3x3'


def test_get_iphone_resolution():
    smc = SMCReader(TEST_SMC_PATH)

    color_resolution = smc.get_iphone_color_resolution(0, vertical=True)
    assert color_resolution.shape == (
        2, ), 'iPhone Color Resolution should be a 2D matrix'
    assert tuple(color_resolution) == (1440, 1920)

    color_resolution = smc.get_iphone_color_resolution(0, vertical=False)
    assert color_resolution.shape == (
        2, ), 'iPhone Color Resolution should be a 2D matrix'
    assert tuple(color_resolution) == (1920, 1440)


def test_get_iphone_color():
    smc = SMCReader(TEST_SMC_PATH)

    with pytest.raises(AssertionError):
        _ = smc.get_color(device='iphone', device_id=0, frame_id=0)

    with pytest.raises(KeyError):
        _ = smc.get_color(device='iPhone', device_id=1, frame_id=0)

    with pytest.raises(TypeError):
        _ = smc.get_color(device='iPhone', device_id=0, frame_id=0.0)

    iphone_color = smc.get_iphone_color(iphone_id=0, frame_id=0, vertical=True)
    assert iphone_color.shape == (1, 1920, 1440, 3), \
        'iPhone color image in vertical mode ' \
        'should have shape 1x1920x1440x3.'

    iphone_color = smc.get_iphone_color(
        iphone_id=0, frame_id=0, vertical=False)
    assert iphone_color.shape == (1, 1440, 1920, 3), \
        'iPhone color image in horizontal mode ' \
        'should have shape 1x1440x1920x3.'


def test_get_iphone_depth():
    smc = SMCReader(TEST_SMC_PATH)

    with pytest.raises(KeyError):
        _ = smc.get_iphone_depth(iphone_id=1, frame_id=0)

    iphone_depth = smc.get_iphone_depth(iphone_id=0, frame_id=0, vertical=True)
    assert iphone_depth.shape == (1, 256, 192), \
        'iPhone depth image in vertical mode should have shape 1x256x192.'

    iphone_depth = smc.get_iphone_depth(
        iphone_id=0, frame_id=0, vertical=False)
    assert iphone_depth.shape == (1, 192, 256), \
        'iPhone depth image in horizontal mode should have shape 1x192x256.'


def test_get_kinect_color():
    smc = SMCReader(TEST_SMC_PATH)

    with pytest.raises(AssertionError):
        _ = smc.get_color(device='kinect', device_id=0, frame_id=0)

    with pytest.raises(KeyError):
        _ = smc.get_color(device='Kinect', device_id=10, frame_id=0)

    with pytest.raises(TypeError):
        _ = smc.get_color(device='Kinect', device_id=0, frame_id=0.0)

    kinect_color = smc.get_color(device='Kinect', device_id=0, frame_id=0)
    assert kinect_color.shape == (1, 1080, 1920, 3), \
        'Kinect Color should have resolution of 1x1080x1920x3'

    kinect_color = smc.get_kinect_color(0, frame_id=0)
    assert kinect_color.shape == (1, 1080, 1920, 3), \
        'Kinect Color should have resolution of 1x1080x1920x3'


def test_get_kinect_depth():
    smc = SMCReader(TEST_SMC_PATH)

    kinect_depth = smc.get_kinect_depth(0, frame_id=0)
    assert kinect_depth.shape == (1, 576, 640), \
        'Kinect depth should have resolution of 1x576x640'


def test_get_kinect_rgbd():
    smc = SMCReader(TEST_SMC_PATH)

    mapped_color, depth = smc.get_kinect_rgbd(0, 0)
    assert mapped_color.shape == (576, 640, 3), \
        'Kinect RGBD should have color resolution of 576x640x3'


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


def test_get_kinect_keypoints2d():
    smc = SMCReader(TEST_SMC_PATH)

    with pytest.raises(AssertionError):
        _ = smc.get_kinect_keypoints2d(device_id=-1)

    keypoints2d, keypoints2d_mask = smc.get_kinect_keypoints2d(device_id=1)
    keypoints_num_frames = smc.get_keypoints_num_frames()
    keypoints_convention = smc.get_keypoints_convention()
    assert keypoints2d.shape[1] == keypoints2d_mask.shape[0]
    assert keypoints2d.shape[0] == keypoints_num_frames
    assert keypoints_convention == 'coco_wholebody'
    assert keypoints2d.shape == (1, 133, 3)
    assert keypoints2d_mask.shape == (133, )
    assert isinstance(keypoints2d, np.ndarray)
    assert isinstance(keypoints2d_mask, np.ndarray)


def test_get_kinect_keypoints2d_by_frame():
    smc = SMCReader(TEST_SMC_PATH)

    keypoints2d, keypoints2d_mask = smc.get_kinect_keypoints2d(
        device_id=1, frame_id=0)
    assert keypoints2d.shape == (1, 133, 3)
    assert keypoints2d_mask.shape == (133, )
    assert isinstance(keypoints2d, np.ndarray)
    assert isinstance(keypoints2d_mask, np.ndarray)


def test_get_kinect_keypoints2d_by_frames():
    smc = SMCReader(TEST_SMC_PATH)

    keypoints2d, keypoints2d_mask = smc.get_kinect_keypoints2d(
        device_id=1, frame_id=[0])
    assert keypoints2d.shape == (1, 133, 3)
    assert keypoints2d_mask.shape == (133, )
    assert isinstance(keypoints2d, np.ndarray)
    assert isinstance(keypoints2d_mask, np.ndarray)

    with pytest.raises(AssertionError):
        keypoints2d, keypoints2d_mask = smc.get_kinect_keypoints2d(
            device_id=100)


def test_get_iphone_keypoints2d():
    smc = SMCReader(TEST_SMC_PATH)

    with pytest.raises(AssertionError):
        _ = smc.get_iphone_keypoints2d(device_id=-1)

    keypoints2d, keypoints2d_mask = smc.get_iphone_keypoints2d(device_id=0)
    keypoints_num_frames = smc.get_keypoints_num_frames()
    keypoints_convention = smc.get_keypoints_convention()
    assert keypoints2d.shape[1] == keypoints2d_mask.shape[0]
    assert keypoints2d.shape[0] == keypoints_num_frames
    assert keypoints_convention == 'coco_wholebody'
    assert keypoints2d.shape == (1, 133, 3)
    assert keypoints2d_mask.shape == (133, )
    assert isinstance(keypoints2d, np.ndarray)
    assert isinstance(keypoints2d_mask, np.ndarray)


def test_get_iphone_keypoints2d_by_frame():
    smc = SMCReader(TEST_SMC_PATH)

    keypoints2d, keypoints2d_mask = smc.get_iphone_keypoints2d(
        device_id=0, frame_id=0)
    assert keypoints2d.shape == (1, 133, 3)
    assert keypoints2d_mask.shape == (133, )
    assert isinstance(keypoints2d, np.ndarray)
    assert isinstance(keypoints2d_mask, np.ndarray)


def test_get_iphone_keypoints2d_by_frames():
    smc = SMCReader(TEST_SMC_PATH)

    keypoints2d, keypoints2d_mask = smc.get_iphone_keypoints2d(
        device_id=0, frame_id=[0])
    assert keypoints2d.ndim == 3
    assert keypoints2d_mask.ndim == 1
    assert keypoints2d.shape == (1, 133, 3)
    assert keypoints2d_mask.shape == (133, )
    assert isinstance(keypoints2d, np.ndarray)
    assert isinstance(keypoints2d_mask, np.ndarray)


def test_get_all_keypoints3d():
    smc = SMCReader(TEST_SMC_PATH)

    # test get all keypoints3d
    keypoints3d, keypoints3d_mask = smc.get_keypoints3d()
    keypoints_num_frames = smc.get_keypoints_num_frames()
    keypoints_convention = smc.get_keypoints_convention()
    keypoints_created_time = smc.get_keypoints_created_time()
    assert keypoints3d.shape[1] == keypoints3d_mask.shape[0]
    assert keypoints3d.shape[0] == keypoints_num_frames
    assert keypoints_convention == 'coco_wholebody'
    assert keypoints3d.shape == (1, 133, 4)
    assert keypoints3d_mask.shape == (133, )
    assert isinstance(keypoints_created_time, str)
    assert isinstance(keypoints3d, np.ndarray)
    assert isinstance(keypoints3d_mask, np.ndarray)


def test_get_keypoints3d_by_frame():
    smc = SMCReader(TEST_SMC_PATH)

    keypoints3d, keypoints3d_mask = smc.get_keypoints3d(frame_id=0)
    assert keypoints3d.shape == (1, 133, 4)
    assert keypoints3d_mask.shape == (133, )
    assert isinstance(keypoints3d, np.ndarray)
    assert isinstance(keypoints3d_mask, np.ndarray)


def test_get_keypoints3d_by_device():
    smc = SMCReader(TEST_SMC_PATH)

    # get all
    with pytest.raises(AssertionError):
        _ = smc.get_keypoints3d(device='kinect', device_id=0)
    with pytest.raises(AssertionError):
        _ = smc.get_keypoints3d(device='Kinect', device_id=-1)
    with pytest.raises(KeyError):
        _ = smc.get_keypoints3d(device='Kinect', device_id=10)
    with pytest.raises(TypeError):
        _ = smc.get_keypoints3d(device='Kinect', device_id=0, frame_id=0.0)

    # get by frame_id
    with pytest.raises(AssertionError):
        _ = smc.get_keypoints3d(device='iphone', device_id=0)
    with pytest.raises(AssertionError):
        _ = smc.get_keypoints3d(device='iPhone', device_id=-1)
    with pytest.raises(KeyError):
        _ = smc.get_keypoints3d(device='iPhone', device_id=10)
    with pytest.raises(TypeError):
        _ = smc.get_keypoints3d(device='iPhone', device_id=0, frame_id=0.0)

    keypoints3d, keypoints3d_mask = \
        smc.get_keypoints3d(device='iPhone', device_id=0, frame_id=0)
    assert keypoints3d.shape == (1, 133, 4)
    assert keypoints3d_mask.shape == (133, )
    assert isinstance(keypoints3d, np.ndarray)
    assert isinstance(keypoints3d_mask, np.ndarray)


def test_get_keypoints3d_by_frames():
    smc = SMCReader(TEST_SMC_PATH)

    keypoints3d, keypoints3d_mask = smc.get_keypoints3d(frame_id=[0])
    assert keypoints3d.shape == (1, 133, 4)
    assert keypoints3d_mask.shape == (133, )
    assert isinstance(keypoints3d, np.ndarray)
    assert isinstance(keypoints3d_mask, np.ndarray)


def test_get_all_smpl():
    smc = SMCReader(TEST_SMC_PATH)

    smpl = smc.get_smpl()
    smpl_num_frames = smc.get_smpl_num_frames()
    smpl_created_time = smc.get_smpl_created_time()
    global_orient = smpl['global_orient']
    body_pose = smpl['body_pose']
    transl = smpl['transl']
    betas = smpl['betas']
    assert global_orient.shape == (smpl_num_frames, 3)
    assert body_pose.shape == (smpl_num_frames, 69)
    assert transl.shape == (smpl_num_frames, 3)
    assert betas.shape == (1, 10)
    assert isinstance(smpl_created_time, str)
    assert isinstance(global_orient, np.ndarray)
    assert isinstance(body_pose, np.ndarray)
    assert isinstance(transl, np.ndarray)
    assert isinstance(betas, np.ndarray)


def test_get_smpl_by_frame():
    smc = SMCReader(TEST_SMC_PATH)

    smpl = smc.get_smpl(frame_id=0)
    global_orient = smpl['global_orient']
    body_pose = smpl['body_pose']
    transl = smpl['transl']
    betas = smpl['betas']
    assert global_orient.shape == (1, 3)
    assert body_pose.shape == (1, 69)
    assert transl.shape == (1, 3)
    assert betas.shape == (1, 10)
    assert isinstance(global_orient, np.ndarray)
    assert isinstance(body_pose, np.ndarray)
    assert isinstance(transl, np.ndarray)
    assert isinstance(betas, np.ndarray)


def test_get_smpl_by_device():
    smc = SMCReader(TEST_SMC_PATH)

    with pytest.raises(AssertionError):
        _ = smc.get_smpl(device='kinect', device_id=0)
    with pytest.raises(AssertionError):
        _ = smc.get_smpl(device='Kinect', device_id=-1)
    with pytest.raises(KeyError):
        _ = smc.get_smpl(device='Kinect', device_id=10)
    with pytest.raises(TypeError):
        _ = smc.get_smpl(device='Kinect', device_id=0, frame_id=0.0)

    with pytest.raises(AssertionError):
        _ = smc.get_smpl(device='iphone', device_id=0)
    with pytest.raises(AssertionError):
        _ = smc.get_smpl(device='iPhone', device_id=-1)
    with pytest.raises(KeyError):
        _ = smc.get_smpl(device='iPhone', device_id=10)
    with pytest.raises(TypeError):
        _ = smc.get_smpl(device='iPhone', device_id=0, frame_id=0.0)

    smpl = smc.get_smpl(device='Kinect', device_id=0)
    smpl_num_frames = smc.get_smpl_num_frames()
    smpl_created_time = smc.get_smpl_created_time()
    global_orient = smpl['global_orient']
    body_pose = smpl['body_pose']
    transl = smpl['transl']
    betas = smpl['betas']
    assert global_orient.shape == (smpl_num_frames, 3)
    assert body_pose.shape == (smpl_num_frames, 69)
    assert transl.shape == (smpl_num_frames, 3)
    assert betas.shape == (1, 10)
    assert isinstance(smpl_created_time, str)
    assert isinstance(global_orient, np.ndarray)
    assert isinstance(body_pose, np.ndarray)
    assert isinstance(transl, np.ndarray)
    assert isinstance(betas, np.ndarray)


def test_iphone_rotation():
    smc = SMCReader(TEST_SMC_PATH)

    # get keypoints3d in world coordinate
    keypoints3d, _ = smc.get_keypoints3d(frame_id=0)
    keypoints3d = keypoints3d.squeeze()
    keypoints3d, conf = keypoints3d[:, :3], keypoints3d[:, 3]

    # get intrinsics in vertical mode
    intrinsics = smc.get_iphone_intrinsics(vertical=True)

    # get extrinsics in vertical mode
    cam2world = \
        smc.get_iphone_extrinsics(
            homogeneous=True, vertical=True)
    extrinsics = np.linalg.inv(cam2world)

    # transform keypoints3d to vertical iPhone
    keypoints3d = np.concatenate(
        [keypoints3d, np.ones([*keypoints3d.shape[:-1], 1])],
        axis=-1)  # homogeneous
    keypoints3d = (extrinsics @ keypoints3d.T).T
    keypoints3d = keypoints3d[..., :3]

    # project keypoints3d to keypoints2d on vertical iPhone
    keypoints2d = (intrinsics @ keypoints3d.T).T
    keypoints2d = keypoints2d[..., :2] / keypoints2d[..., [-1]]

    # check validity
    keypoints2d_vertical, _ = smc.get_iphone_keypoints2d(vertical=True)
    keypoints2d_vertical = keypoints2d_vertical.squeeze()[..., :2]
    keypoints2d[conf == 0.0] = 0.0
    assert np.allclose(keypoints2d, keypoints2d_vertical)

    # rotate vertical keypoints2d back to horizontal
    # counter-clockwise by 90 degrees
    W, H = smc.get_iphone_color_resolution(vertical=True)
    xs, ys = keypoints2d[..., 0], keypoints2d[..., 1]
    xs, ys = ys, W - xs  # vertical -> horizontal
    keypoints2d[..., 0], keypoints2d[..., 1] = xs.copy(), ys.copy()

    # check validity
    keypoints2d_horizontal, _ = smc.get_iphone_keypoints2d(vertical=False)
    keypoints2d_horizontal = keypoints2d_horizontal.squeeze()[..., :2]
    keypoints2d[conf == 0.0] = 0.0
    assert np.allclose(keypoints2d, keypoints2d_horizontal)
