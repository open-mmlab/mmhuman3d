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


def test_get_color():
    smc = SMCReader(TEST_SMC_PATH)

    kinect_color = smc.get_color(device='Kinect', device_id=0, frame_id=0)
    assert kinect_color.shape == (
        1, 1080, 1920,
        3), 'Kinect Color should have resolution of 1x1080x1920x3'

    iphone_color = smc.get_color(device='iPhone', device_id=0, frame_id=0)
    assert iphone_color.shape == (
        1, 1440, 1920, 3), 'iPhone color image should have shape 1x1440x1920x3'

    with pytest.raises(AssertionError):
        _ = smc.get_color(device='kinect', device_id=0, frame_id=0)

    with pytest.raises(AssertionError):
        _ = smc.get_color(device='iphone', device_id=0, frame_id=0)

    with pytest.raises(KeyError):
        _ = smc.get_color(device='Kinect', device_id=10, frame_id=0)

    with pytest.raises(KeyError):
        _ = smc.get_color(device='iPhone', device_id=1, frame_id=0)


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
        _ = smc.get_keypoints3d(device='kinect', device_id=10)
    with pytest.raises(AssertionError):
        _ = smc.get_keypoints3d(device='Kinect', device_id=-1)
    with pytest.raises(KeyError):
        _ = smc.get_keypoints3d(device='Kinect', device_id=10)

    # get by frame_id
    with pytest.raises(AssertionError):
        _ = smc.get_keypoints3d(device='iPhone', device_id=-1)
    with pytest.raises(AssertionError):
        _ = smc.get_keypoints3d(device='iPhone', device_id=10)
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
    assert body_pose.shape[0] == smpl_num_frames
    assert global_orient.shape == (1, 3)
    assert body_pose.shape == (1, 69)
    assert transl.shape == (1, 3)
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

smc_reader = SMCReader('/home/caizhongang/github/zoehuman/mmhuman3d/tests/data/dataset_sample/humman/p000003_a000014_tiny.smc')
K = smc_reader.get_iphone_intrinsics()

fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
# K = np.array([
#     [fx, 0, 0, cx],
#     [0, fy, 0, cy],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ])
K = np.array([
    [fx, 0, cx, 0],
    [0, fy, cy, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
K_3x3 = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
])
T = smc_reader.get_iphone_extrinsics(homogeneous=True)
T = np.linalg.inv(T)  # world2cam
# import pdb; pdb.set_trace()
xmax, ymax = 1920, 1440
r = np.eye(4)
r[:2, :2] = np.array([[0,-1],[1,0]])

K_ = np.array([
    [fy, 0, ymax-cy, 0],
    [0, fx, cx, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
K__3x3 = np.array([
    [fy, 0, ymax-cy],
    [0, fx, cx],
    [0, 0, 1]
])
P = K @ r @ T
T_ = r @ T

keypoints3d, keypoints3d_mask = smc_reader.get_keypoints3d(frame_id=0)
keypoints3d = keypoints3d.squeeze()[:, :3]
keypoints3d = np.concatenate([keypoints3d, np.ones([*keypoints3d.shape[:-1], 1])], axis=-1)

import cv2
img = smc_reader.get_color('iPhone', device_id=0, frame_id=0)
img = img.squeeze()
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.namedWindow('test', cv2.WINDOW_NORMAL)

# -------- original iphone

keypoints3d_iphone = (T @ keypoints3d.T).T
keypoints3d_iphone = keypoints3d_iphone[..., :3]

keypoints2d_iphone = (K_3x3 @ keypoints3d_iphone.T).T
keypoints2d_iphone = keypoints2d_iphone[..., :2] / keypoints2d_iphone[..., [-1]]

img_iphone = img.copy()
for keypoint in keypoints2d_iphone:
    x, y = keypoint
    cv2.circle(img_iphone, (int(x), int(y)), radius=5, color=(0,0,255), thickness=-1)

cv2.imshow('test', img_iphone)
cv2.waitKey(0)

# -------- rotated with our derivation

keypoints3d_new = (T_ @ keypoints3d.T).T
keypoints3d_new = keypoints3d_new[..., :3]

keypoints2d_new = (K__3x3 @ keypoints3d_new.T).T
keypoints2d_new = keypoints2d_new[..., :2] / keypoints2d_new[..., [-1]]

img_new = img.copy()
img_new = cv2.rotate(img_new, cv2.cv2.ROTATE_90_CLOCKWISE)
for keypoint in keypoints2d_new:
    x, y = keypoint
    cv2.circle(img_new, (int(x), int(y)), radius=5, color=(0,0,255), thickness=-1)

cv2.imshow('test', img_new)
cv2.waitKey(0)

# -------- test if r is correct
#
# keypoints2d_new = (P @ keypoints3d.T).T
# keypoints2d_new = keypoints2d_new[..., :2] / keypoints2d_new[..., [2]]
#
# img_new = img.copy()
# img_new = cv2.rotate(img_new, cv2.cv2.ROTATE_90_CLOCKWISE)
# for keypoint in keypoints2d_new:
#     x, y = keypoint
#     print(x, y)
#     cv2.circle(img_new, (int(x), int(y)), radius=5, color=(0,0,255), thickness=-1)
#
# cv2.imshow('test', img_new)
# cv2.waitKey(0)
# cv2.destroyWindow('test')