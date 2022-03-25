import os

import numpy as np
import pytest

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d
from mmhuman3d.utils.camera_utils import project_points
from mmhuman3d.utils.ffmpeg_utils import (
    array_to_images,
    images_to_array,
    video_to_array,
)

data_root = 'tests/data/test_vis_kp2d'
os.makedirs(data_root, exist_ok=True)


def test_vis_kp2d():
    image_array = np.random.randint(
        low=0, high=255, size=(2, 512, 512, 3), dtype=np.uint8)
    array_to_images(image_array, data_root, img_format='%06d.png')

    # wrong input shape
    kp2d = np.random.randint(low=0, high=255, size=(133, 2), dtype=np.uint8)
    with pytest.raises(AssertionError):
        visualize_kp2d(
            kp2d,
            output_path='tests/data/test_vis_kp2d/wrong_input_shape.mp4',
            frame_list=[
                'tests/data/test_vis_kp2d/%06d.png' % 1,
                'tests/data/test_vis_kp2d/%06d.png' % 2
            ],
        )

    # wrong input frame path
    kp2d = np.random.randint(
        low=0, high=255, size=(10, 133, 2), dtype=np.uint8)
    with pytest.raises(FileNotFoundError):
        visualize_kp2d(
            kp2d,
            output_path='tests/data/test_vis_kp2d/wrong_input_path.mp4',
            frame_list=[
                'tests/data/test_vis_kp2d/1.png',
                'tests/data/test_vis_kp2d/2.png'
            ])

    # wrong pop parts
    kp2d = np.random.randint(
        low=0, high=255, size=(10, 133, 2), dtype=np.uint8)
    with pytest.raises(AssertionError):
        visualize_kp2d(
            kp2d,
            output_path='tests/data/test_vis_kp2d/wrong_pop_parts.mp4',
            frame_list=[
                'tests/data/test_vis_kp2d/%06d.png' % 1,
                'tests/data/test_vis_kp2d/%06d.png' % 2
            ],
            pop_parts=['rubbish'],
        )

    # wrong data_source
    kp2d = np.random.randint(low=0, high=255, size=(10, 17, 3), dtype=np.uint8)
    with pytest.raises(IndexError):
        visualize_kp2d(
            kp2d,
            output_path='tests/data/test_vis_kp2d/wrong_data_source.mp4',
            frame_list=[
                'tests/data/test_vis_kp2d/%06d.png' % 0,
                'tests/data/test_vis_kp2d/%06d.png' % 1
            ],
            data_source='coco_wholebody',
        )

    # test shape
    kp2d = np.random.randint(low=0, high=16, size=(10, 133, 2), dtype=np.uint8)
    visualize_kp2d(
        kp2d,
        output_path='tests/data/test_vis_kp2d/test_shape.mp4',
        frame_list=[
            'tests/data/test_vis_kp2d/%06d.png' % 0,
            'tests/data/test_vis_kp2d/%06d.png' % 1
        ],
        overwrite=True,
    )
    assert video_to_array('tests/data/test_vis_kp2d/test_shape.mp4').shape

    # file exists
    with pytest.raises(FileExistsError):
        kp2d = np.random.randint(
            low=0, high=16, size=(10, 133, 2), dtype=np.uint8)
        visualize_kp2d(
            kp2d,
            output_path='tests/data/test_vis_kp2d/test_shape.mp4',
            frame_list=[
                'tests/data/test_vis_kp2d/%06d.png' % 0,
                'tests/data/test_vis_kp2d/%06d.png' % 1
            ],
            overwrite=False,
        )

    # test multi-person shape
    kp2d = np.random.randint(
        low=0, high=255, size=(10, 2, 133, 2), dtype=np.uint8)
    visualize_kp2d(
        kp2d,
        output_path='tests/data/test_vis_kp2d/test_multi.mp4',
        frame_list=[
            'tests/data/test_vis_kp2d/%06d.png' % 0,
            'tests/data/test_vis_kp2d/%06d.png' % 1
        ],
        overwrite=True,
    )
    assert video_to_array('tests/data/test_vis_kp2d/test_multi.mp4').shape

    # test shape with confidence
    kp2d = np.random.randint(
        low=0, high=255, size=(10, 133, 3), dtype=np.uint8)
    visualize_kp2d(
        kp2d,
        output_path='tests/data/test_vis_kp2d/test_confidence.mp4',
        frame_list=[
            'tests/data/test_vis_kp2d/%06d.png' % 0,
            'tests/data/test_vis_kp2d/%06d.png' % 1
        ],
        overwrite=True,
    )
    assert video_to_array('tests/data/test_vis_kp2d/test_confidence.mp4').shape

    visualize_kp2d(
        kp2d,
        output_path='tests/data/test_vis_kp2d/test_confidence1.mp4',
        origin_frames='tests/data/test_vis_kp2d/test_confidence.mp4',
        overwrite=True,
    )
    assert video_to_array(
        'tests/data/test_vis_kp2d/test_confidence1.mp4').shape

    # visualize single frame
    kp2d = np.random.randint(low=0, high=255, size=(1, 17, 2), dtype=np.uint8)
    image_array = visualize_kp2d(
        kp2d,
        image_array=image_array,
        data_source='coco',
        draw_bbox=True,
        with_number=True,
        return_array=True)
    assert isinstance(image_array,
                      np.ndarray) and image_array.shape == (1, 512, 512, 3)

    # visualize single frame
    kp2d = np.random.randint(low=0, high=255, size=(1, 17, 2), dtype=np.uint8)
    kp2d, mask = convert_kps(keypoints=kp2d, src='coco', dst='coco_wholebody')
    image_array = visualize_kp2d(
        kp2d,
        image_array=image_array,
        data_source='coco_wholebody',
        mask=mask,
        draw_bbox=True,
        with_number=True,
        return_array=True)
    assert isinstance(image_array,
                      np.ndarray) and image_array.shape == (1, 512, 512, 3)

    # test output folder
    output_folder = 'tests/data/test_vis_kp2d/1/'
    kp2d = np.random.randint(low=0, high=16, size=(10, 133, 2), dtype=np.uint8)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    visualize_kp2d(
        kp2d,
        output_path=output_folder,
        frame_list=[
            'tests/data/test_vis_kp2d/%06d.png' % 0,
            'tests/data/test_vis_kp2d/%06d.png' % 1
        ],
        overwrite=True,
    )
    assert images_to_array(output_folder).shape

    # test output folder same as input folder
    output_folder = 'tests/data/test_vis_kp2d/1/'
    kp2d = np.random.randint(low=0, high=16, size=(10, 133, 2), dtype=np.uint8)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with pytest.raises(FileExistsError):
        visualize_kp2d(
            kp2d,
            output_path=output_folder,
            frame_list=[
                os.path.join(output_folder, '%06d.png' % 0),
                os.path.join(output_folder, '%06d.png' % 1)
            ],
            overwrite=False,
        )

    # test project kp3d to kp2d
    output_folder = 'tests/data/test_vis_kp2d/1/'
    kp3d = np.random.uniform(size=(10, 133, 3), low=0, high=10)
    cameras = build_cameras(dict(type='perspective'))
    kp2d = project_points(kp3d, cameras, resolution=(1000, 1000))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    visualize_kp2d(
        kp2d,
        output_path=output_folder,
        frame_list=[
            'tests/data/test_vis_kp2d/%06d.png' % 0,
            'tests/data/test_vis_kp2d/%06d.png' % 1
        ],
        overwrite=True,
    )
    assert images_to_array(output_folder).shape

    # img_format
    output_folder = 'tests/data/test_vis_kp2d/1/'
    kp2d = np.random.randint(low=0, high=16, size=(10, 133, 2), dtype=np.uint8)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    visualize_kp2d(
        kp2d,
        output_path=output_folder,
        frame_list=[
            'tests/data/test_vis_kp2d/%06d.png' % 0,
            'tests/data/test_vis_kp2d/%06d.png' % 1
        ],
        img_format=None,
        overwrite=True,
    )
    assert images_to_array(output_folder).shape

    visualize_kp2d(
        kp2d,
        output_path=output_folder,
        origin_frames='tests/data/test_vis_kp2d/',
        img_format='%06d.png',
        overwrite=True,
        resolution=(1000, 1000),
    )
    assert images_to_array(output_folder).shape
