import os

import numpy as np
import pytest

from mmhuman3d.core.conventions.joints_mapping.kp_mapping import convert_kps
from mmhuman3d.core.visualization.ffmpeg_utils import (
    array_to_images,
    video_to_array,
)
from mmhuman3d.core.visualization.keypoint_utils import search_limbs
from mmhuman3d.core.visualization.visualize_keypoints2d import (
    plot_kp2d_frame,
    render_kp2d_to_video,
)

osj = os.path.join


def test_vis_kp2d():
    image_array = np.random.randint(
        low=0, high=255, size=(2, 512, 512, 3), dtype=np.uint8)
    array_to_images(image_array, '/tmp')

    # wrong input shape
    kp2d = np.random.randint(low=0, high=255, size=(133, 2), dtype=np.uint8)
    with pytest.raises(AssertionError):
        render_kp2d_to_video(
            kp2d,
            output_path='/tmp/1.mp4',
            frame_list=['/tmp/%06d.png' % 1,
                        '/tmp/%06d.png' % 2],
        )

    # wrong input frame path
    kp2d = np.random.randint(
        low=0, high=255, size=(10, 133, 2), dtype=np.uint8)
    with pytest.raises(FileNotFoundError):
        render_kp2d_to_video(
            kp2d,
            output_path='/tmp/1.mp4',
            frame_list=['/tmp/1.png', '/tmp/2.png'])

    # wrong output path
    kp2d = np.random.randint(
        low=0, high=255, size=(10, 133, 2), dtype=np.uint8)
    with pytest.raises(FileNotFoundError):
        render_kp2d_to_video(
            kp2d,
            output_path='/123/1.mp4',
            frame_list=['/tmp/%06d.png' % 1,
                        '/tmp/%06d.png' % 2])

    # wrong pop parts
    kp2d = np.random.randint(
        low=0, high=255, size=(10, 133, 2), dtype=np.uint8)
    with pytest.raises(KeyError):
        render_kp2d_to_video(
            kp2d,
            output_path='/tmp/1.mp4',
            frame_list=['/tmp/%06d.png' % 1,
                        '/tmp/%06d.png' % 2],
            pop=['rubbish'])

    # wrong data_source
    kp2d = np.random.randint(low=0, high=255, size=(10, 17, 3), dtype=np.uint8)
    with pytest.raises(IndexError):
        render_kp2d_to_video(
            kp2d,
            output_path='/tmp/1.mp4',
            frame_list=['/tmp/%06d.png' % 1,
                        '/tmp/%06d.png' % 2],
            data_source='mmpose',
        )

    # test shape
    kp2d = np.random.randint(low=0, high=16, size=(10, 133, 2), dtype=np.uint8)
    render_kp2d_to_video(
        kp2d,
        output_path='/tmp/1.mp4',
        frame_list=['/tmp/%06d.png' % 1,
                    '/tmp/%06d.png' % 2],
    )
    assert video_to_array('/tmp/1.mp4').shape

    # test multi-person shape
    kp2d = np.random.randint(
        low=0, high=255, size=(10, 2, 133, 2), dtype=np.uint8)
    render_kp2d_to_video(
        kp2d,
        output_path='/tmp/1.mp4',
        frame_list=['/tmp/%06d.png' % 1,
                    '/tmp/%06d.png' % 2],
    )
    assert video_to_array('/tmp/1.mp4').shape

    # test shape with confidence
    kp2d = np.random.randint(
        low=0, high=255, size=(10, 133, 3), dtype=np.uint8)
    render_kp2d_to_video(
        kp2d,
        output_path='/tmp/1.mp4',
        frame_list=['/tmp/%06d.png' % 1,
                    '/tmp/%06d.png' % 2],
    )
    assert video_to_array('/tmp/1.mp4').shape

    # visualize single frame
    kp2d = np.random.randint(low=0, high=255, size=(1, 17, 2), dtype=np.uint8)
    limbs_target, limbs_palette = search_limbs(data_source='coco', mask=None)
    image_array = np.random.randint(
        low=0, high=255, size=(512, 512, 3), dtype=np.uint8)
    image_array = plot_kp2d_frame(
        kp2d[0],
        image_array,
        limbs_target,
        limbs_palette,
        draw_bbox=True,
        with_number=True,
        font_size=0.5)
    assert isinstance(image_array,
                      np.ndarray) and image_array.shape == (512, 512, 3)

    # visualize single frame
    kp2d = np.random.randint(low=0, high=255, size=(1, 17, 2), dtype=np.uint8)
    kp2d, mask = convert_kps(joints=kp2d, src='coco', dst='mmpose')
    limbs_target, limbs_palette = search_limbs(data_source='mmpose', mask=mask)
    image_array = np.random.randint(
        low=0, high=255, size=(512, 512, 3), dtype=np.uint8)
    image_array = plot_kp2d_frame(
        kp2d[0],
        image_array,
        limbs_target,
        limbs_palette,
        draw_bbox=True,
        with_number=True,
        font_size=0.5)
    assert isinstance(image_array,
                      np.ndarray) and image_array.shape == (512, 512, 3)

    # visualize single frame
    with pytest.raises(IndexError):
        kp2d = np.random.randint(low=0, high=255, size=(17, 2), dtype=np.uint8)
        limbs_target, limbs_palette = search_limbs(
            data_source='mmpose', mask=None)
        image_array = np.random.randint(
            low=0, high=255, size=(512, 512, 3), dtype=np.uint8)
        image_array = plot_kp2d_frame(
            kp2d,
            image_array,
            limbs_target,
            limbs_palette,
            draw_bbox=True,
            with_number=True,
            font_size=0.5)
