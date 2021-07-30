import numpy as np
import pytest

from mmhuman3d.core.conventions.joints_mapping.kp_mapping import convert_kps
from mmhuman3d.core.visualization import visualize_keypoints3d
from mmhuman3d.core.visualization.ffmpeg_utils import video_to_array

render_kp3d_to_video = visualize_keypoints3d.render_kp3d_to_video


def test_vis_kp3d():
    # wrong input shape
    with pytest.raises(AssertionError):
        joints = np.random.randint(
            low=0, high=255, size=(133, 3), dtype=np.uint8)
        render_kp3d_to_video(
            joints,
            '/tmp/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(512, 512),
            data_source='mmpose')
    with pytest.raises(AssertionError):
        joints = np.random.randint(
            low=0, high=255, size=(30, 133, 1), dtype=np.uint8)
        render_kp3d_to_video(
            joints,
            '/tmp/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(512, 512),
            data_source='mmpose')

    with pytest.raises(KeyError):
        joints = np.random.randint(
            low=0, high=255, size=(30, 133, 3), dtype=np.uint8)
        render_kp3d_to_video(
            joints,
            '/tmp/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(512, 512),
            data_source='mmpose',
            pop_parts=['rubbish'])

    # wrong output path
    with pytest.raises(FileNotFoundError):
        joints = np.random.randint(
            low=0, high=255, size=(1, 133, 3), dtype=np.uint8)
        render_kp3d_to_video(
            joints,
            '/123/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(512, 512),
            data_source='mmpose')
    with pytest.raises(FileNotFoundError):
        joints = np.random.randint(
            low=0, high=255, size=(1, 133, 3), dtype=np.uint8)
        render_kp3d_to_video(
            joints,
            '/tmp/tmp.mov',
            mask=None,
            orbit_speed=0.5,
            resolution=(512, 512),
            data_source='mmpose')

    joints = np.random.randint(
        low=0, high=255, size=(30, 1, 133, 3), dtype=np.uint8)
    render_kp3d_to_video(
        joints,
        '/tmp/tmp.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(512, 512),
        data_source='mmpose')
    assert video_to_array('/tmp/tmp.mp4').shape

    joints = np.random.randint(
        low=0, high=255, size=(30, 133, 3), dtype=np.uint8)
    render_kp3d_to_video(
        joints,
        '/tmp/tmp.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(512, 512),
        data_source='mmpose')
    assert video_to_array('/tmp/tmp.mp4').shape

    joints = np.random.randint(
        low=0, high=255, size=(30, 2, 133, 3), dtype=np.uint8)
    render_kp3d_to_video(
        joints,
        '/tmp/tmp.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(512, 512),
        data_source='mmpose')
    assert video_to_array('/tmp/tmp.mp4').shape

    joints = np.random.randint(
        low=0, high=255, size=(30, 1, 133, 4), dtype=np.uint8)
    render_kp3d_to_video(
        joints,
        '/tmp/tmp.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(512, 512),
        data_source='mmpose')
    assert video_to_array('/tmp/tmp.mp4').shape

    joints = np.random.randint(
        low=0, high=255, size=(30, 1, 17, 3), dtype=np.uint8)
    joints, mask = convert_kps(joints=joints, src='coco', dst='mmpose')
    assert joints.shape == (30, 1, 133, 3)
    render_kp3d_to_video(
        joints,
        '/tmp/tmp.mp4',
        mask=mask,
        orbit_speed=0.5,
        resolution=(512, 512),
        data_source='mmpose')
    assert video_to_array('/tmp/tmp.mp4').shape
