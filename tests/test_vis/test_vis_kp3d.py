import os
import shutil

import numpy as np
import pytest

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.visualization import visualize_kp3d
from mmhuman3d.utils.ffmpeg_utils import video_to_array

data_root = 'tests/data/test_vis_kp3d'
os.makedirs(data_root, exist_ok=True)


def test_vis_kp3d():
    # wrong input shape
    with pytest.raises(AssertionError):
        keypoints = np.random.randint(
            low=0, high=255, size=(133, 3), dtype=np.uint8)
        visualize_kp3d(
            keypoints,
            'tests/data/test_vis_kp3d/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(128, 128),
            data_source='coco_wholebody')
    with pytest.raises(AssertionError):
        keypoints = np.random.randint(
            low=0, high=255, size=(5, 133, 1), dtype=np.uint8)
        visualize_kp3d(
            keypoints,
            'tests/data/test_vis_kp3d/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(128, 128),
            data_source='coco_wholebody')
    # wrong pop_parts
    with pytest.raises(AssertionError):
        keypoints = np.random.randint(
            low=0, high=255, size=(5, 133, 3), dtype=np.uint8)
        visualize_kp3d(
            keypoints,
            'tests/data/test_vis_kp3d/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(128, 128),
            data_source='coco_wholebody',
            pop_parts=['rubbish'])

    # wrong type
    with pytest.raises(FileNotFoundError):
        keypoints = np.random.randint(
            low=0, high=255, size=(1, 133, 3), dtype=np.uint8)
        visualize_kp3d(
            keypoints,
            'tests/data/test_vis_kp3d/tmp.mov',
            mask=None,
            orbit_speed=0.5,
            resolution=(128, 128),
            data_source='coco_wholebody')
    # wrong keypoints type
    with pytest.raises(TypeError):
        visualize_kp3d(
            keypoints.tolist(),
            'tests/data/test_vis_kp3d/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(128, 128),
            data_source='coco_wholebody')
    # wrong data_source
    with pytest.raises(ValueError):
        visualize_kp3d(
            keypoints,
            'tests/data/test_vis_kp3d/tmp.mp4',
            mask=None,
            orbit_speed=0.5,
            resolution=(128, 128),
            data_source='soso_wholebody')
    # parent dir will be created
    keypoints = np.random.randint(
        low=0, high=255, size=(1, 133, 3), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/123/tmp.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody')
    keypoints = np.random.randint(
        low=0, high=255, size=(5, 133, 3), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/frames_dir',
        mask=None,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody')
    assert len(os.listdir('tests/data/test_vis_kp3d/frames_dir')) > 0

    visualize_kp3d(
        keypoints,
        output_path=None,
        mask=None,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody')

    visualize_kp3d(
        keypoints[..., :2],
        'tests/data/test_vis_kp3d/frame_names_0.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody',
        limbs=[[0, 1]],
        palette=[
            [1, 1, 1],
        ],
        frame_names='%06d.jpg')
    # frame_names = [f'frame_{idx}' for idx in range(5)]
    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/frame_names_1.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody',
        frame_names='coco_title')

    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/pop_parts.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody',
        pop_parts=[
            'mouth',
        ])

    keypoints = np.random.randint(
        low=0, high=255, size=(5, 1, 133, 3), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/dim_4.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody')
    assert video_to_array('tests/data/test_vis_kp3d/dim_4.mp4').shape

    keypoints = np.random.randint(
        low=0, high=255, size=(5, 133, 3), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/single_person.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody')
    assert video_to_array('tests/data/test_vis_kp3d/single_person.mp4').shape

    keypoints = np.random.randint(
        low=0, high=255, size=(5, 2, 133, 3), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/multi_person.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody')
    assert video_to_array('tests/data/test_vis_kp3d/multi_person.mp4').shape

    keypoints = np.random.randint(
        low=0, high=255, size=(5, 1, 133, 4), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/with_conf.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody')
    assert video_to_array('tests/data/test_vis_kp3d/with_conf.mp4').shape

    keypoints = np.random.randint(
        low=0, high=255, size=(5, 1, 17, 3), dtype=np.uint8)
    keypoints, mask = convert_kps(
        keypoints=keypoints, src='coco', dst='coco_wholebody')
    assert keypoints.shape == (5, 1, 133, 3)
    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/with_mask.mp4',
        mask=mask,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody')
    assert video_to_array('tests/data/test_vis_kp3d/with_mask.mp4').shape

    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/with_mask_list.mp4',
        mask=mask.tolist(),
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody')
    assert video_to_array('tests/data/test_vis_kp3d/with_mask_list.mp4').shape


def test_renderer():
    # long video to reach the horizontal border
    keypoints = np.random.randint(
        low=0, high=255, size=(110, 1, 18, 3), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/hori_border.mp4',
        mask=None,
        orbit_speed=20,
        resolution=(16, 16),
        data_source='pw3d')
    assert video_to_array('tests/data/test_vis_kp3d/hori_border.mp4').shape
    # set value_range to None for auto range
    keypoints = np.random.randint(
        low=0, high=255, size=(5, 133, 3), dtype=np.uint8)
    visualize_kp3d(
        keypoints,
        'tests/data/test_vis_kp3d/auto_range.mp4',
        mask=None,
        orbit_speed=0.5,
        resolution=(128, 128),
        data_source='coco_wholebody',
        value_range=None)
    assert video_to_array('tests/data/test_vis_kp3d/auto_range.mp4').shape


def test_end():
    shutil.rmtree(data_root)
