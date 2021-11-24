import numpy as np
import pytest

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.visualization import visualize_kp3d
from mmhuman3d.utils.ffmpeg_utils import video_to_array


def test_vis_kp3d():
    # wrong input shape
    with pytest.raises(AssertionError):
        keypoints = np.random.randint(
            low=0, high=255, size=(133, 3), dtype=np.uint8)
        visualize_kp3d(
            keypoints,
            '/tmp/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(512, 512),
            data_source='coco_wholebody')
    with pytest.raises(AssertionError):
        keypoints = np.random.randint(
            low=0, high=255, size=(30, 133, 1), dtype=np.uint8)
        visualize_kp3d(
            keypoints,
            '/tmp/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(512, 512),
            data_source='coco_wholebody')

    with pytest.raises(AssertionError):
        keypoints = np.random.randint(
            low=0, high=255, size=(30, 133, 3), dtype=np.uint8)
        visualize_kp3d(
            keypoints,
            '/tmp/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(512, 512),
            data_source='coco_wholebody',
            pop_parts=['rubbish'])

    # wrong output path
    with pytest.raises(FileNotFoundError):
        keypoints = np.random.randint(
            low=0, high=255, size=(1, 133, 3), dtype=np.uint8)
        visualize_kp3d(
            keypoints,
            '/tmp/tmp.mov',
            mask=None,
            orbit_speed=0.5,
            resolution=(512, 512),
            data_source='coco_wholebody')

    keypoints = np.random.randint(
        low=0, high=255, size=(30, 1, 133, 3), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        '/tmp/tmp.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(512, 512),
        data_source='coco_wholebody')
    assert video_to_array('/tmp/tmp.mp4').shape

    keypoints = np.random.randint(
        low=0, high=255, size=(30, 133, 3), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        '/tmp/tmp.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(512, 512),
        data_source='coco_wholebody')
    assert video_to_array('/tmp/tmp.mp4').shape

    keypoints = np.random.randint(
        low=0, high=255, size=(30, 2, 133, 3), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        '/tmp/tmp.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(512, 512),
        data_source='coco_wholebody')
    assert video_to_array('/tmp/tmp.mp4').shape

    keypoints = np.random.randint(
        low=0, high=255, size=(30, 1, 133, 4), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        '/tmp/tmp.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(512, 512),
        data_source='coco_wholebody')
    assert video_to_array('/tmp/tmp.mp4').shape

    keypoints = np.random.randint(
        low=0, high=255, size=(30, 1, 17, 3), dtype=np.uint8)
    keypoints, mask = convert_kps(
        keypoints=keypoints, src='coco', dst='coco_wholebody')
    assert keypoints.shape == (30, 1, 133, 3)
    visualize_kp3d(
        keypoints,
        '/tmp/tmp.mp4',
        mask=mask,
        orbit_speed=0.5,
        resolution=(512, 512),
        data_source='coco_wholebody')
    assert video_to_array('/tmp/tmp.mp4').shape
