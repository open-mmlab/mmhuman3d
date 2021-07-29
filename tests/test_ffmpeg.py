import os
import os.path as osp

import numpy as np
import pytest

from mmhuman3d.core.visualization.ffmpeg_utils import (
    array_to_images,
    array_to_video,
    compress_video,
    gif_to_images,
    gif_to_video,
    images_to_array,
    images_to_gif,
    images_to_video,
    spatial_concat_video,
    spatial_crop_video,
    temporal_concat_video,
    temporal_crop_video,
    vid_info,
    video_to_array,
    video_to_gif,
    video_to_images,
)

os.makedirs('/tmp/ffmpeg_test', exist_ok=True)
root = '/tmp/ffmpeg_test'


def test_array_saver():
    # not array
    with pytest.raises(TypeError):
        v = None
        array_to_video(v, '/tmp/test.mp4')

    # wrong shape
    with pytest.raises(AssertionError):
        v = np.random.randint(
            low=0, high=255, size=(30, 512, 512, 4), dtype=np.uint8)
        array_to_video(v, osp.join(root, 'test.mp4'))
        os.makedirs(osp.join(root, 'img_folder/'))
        array_to_images(v, osp.join(root, 'img_folder/'))
    with pytest.raises(AssertionError):
        v = np.random.randint(
            low=0, high=255, size=(512, 512, 3), dtype=np.uint8)
        array_to_video(v, osp.join(root, 'test.mp4'))
        array_to_images(v, osp.join(root, 'img_folder/'))

    # GIF not supported
    with pytest.raises(FileNotFoundError):
        v = np.random.randint(
            low=0, high=255, size=(30, 512, 512, 3), dtype=np.uint8)
        array_to_video(v, osp.join(root, 'test.gif'))

    # wrong out path
    with pytest.raises(FileNotFoundError):
        v = np.random.randint(
            low=0, high=255, size=(30, 512, 512, 3), dtype=np.uint8)
        array_to_video(v, '/123/test.mp4')
        array_to_images(v, osp.join('/123/', 'img_folder/'))

    v = np.random.randint(
        low=0, high=255, size=(30, 512, 512, 3), dtype=np.uint8)
    array_to_images(v, osp.join(root, 'img_folder/'))
    array_to_video(v, osp.join(root, 'test.mp4'))
    assert os.path.isfile(osp.join(root, 'test.mp4'))
    assert os.path.isfile(osp.join(root, 'img_folder/', '%06d.png' % 1))


def test_image_reader():
    # images_to_array, video_to_array, vid_info
    v = np.random.randint(
        low=0, high=255, size=(30, 512, 512, 3), dtype=np.uint8)
    array_to_images(v, osp.join(root, 'img_folder/'))
    array_to_video(v, osp.join(root, 'test.mp4'))
    # wrong path
    with pytest.raises(FileNotFoundError):
        _ = images_to_array(osp.join('123', 'img_folder/'))

    # shape should be (f, h, w, 3)
    v = images_to_array(osp.join(root, 'img_folder/'), resolution=(300, 200))
    assert v.shape[1:] == (200, 300, 3)
    v = video_to_array(osp.join(root, 'test.mp4'), resolution=(300, 200))
    assert v.shape[1:] == (200, 300, 3)

    vid = vid_info(osp.join(root, 'test.mp4'))
    for k in [
            'index', 'codec_name', 'codec_long_name', 'profile', 'codec_type',
            'codec_time_base', 'codec_tag_string', 'codec_tag', 'width',
            'height', 'coded_width', 'coded_height', 'has_b_frames', 'pix_fmt',
            'level', 'chroma_location', 'refs', 'is_avc', 'nal_length_size',
            'r_frame_rate', 'avg_frame_rate', 'time_base', 'start_pts',
            'start_time', 'duration_ts', 'duration', 'bit_rate',
            'bits_per_raw_sample', 'nb_frames', 'disposition', 'tags'
    ]:
        assert k in vid.video_stream


def test_convert():
    # images_to_gif, images_to_video, gif_to_video,
    # gif_to_images, video_to_gif, video_to_images

    # wrong inpath images_to_gif
    v = np.random.randint(
        low=0, high=255, size=(30, 512, 512, 3), dtype=np.uint8)
    array_to_images(v, osp.join(root, 'img_folder/'))

    with pytest.raises(FileNotFoundError):
        images_to_gif(
            osp.join('123', 'img_folder'), osp.join(root, 'test.gif'))
    # wrong outpath
    with pytest.raises(FileNotFoundError):
        images_to_gif(
            osp.join(root, 'img_folder'), osp.join('123', 'test.gif'))
    images_to_gif(
        osp.join(root, 'img_folder'), osp.join(root, 'test.gif'), fps=30)
    assert os.path.isfile(osp.join(root, 'test.gif'))

    # wrong inpath images_to_video
    with pytest.raises(FileNotFoundError):
        images_to_video(
            osp.join('123', 'img_folder'), osp.join(root, 'test.mp4'))
    # wrong outpath
    with pytest.raises(FileNotFoundError):
        images_to_video(
            osp.join(root, 'img_folder'), osp.join('123', 'test.mp4'))
    images_to_video(osp.join(root, 'img_folder'), osp.join(root, 'test.mp4'))
    assert os.path.isfile(osp.join(root, 'test.mp4'))

    # wrong inpath gif_to_video
    with pytest.raises(FileNotFoundError):
        gif_to_video(osp.join('123', 'test.gif'), osp.join(root, 'test.mp4'))
    # wrong outpath
    with pytest.raises(FileNotFoundError):
        gif_to_video(osp.join(root, 'test.gif'), osp.join('123', 'test.mp4'))
    gif_to_video(osp.join(root, 'test.gif'), osp.join(root, 'test.mp4'))
    assert os.path.isfile(osp.join(root, 'test.mp4'))

    # wrong inpath gif_to_images
    with pytest.raises(FileNotFoundError):
        gif_to_images(
            osp.join('123', 'test.gif'), osp.join(root, 'img_folder'))
    # wrong outpath
    with pytest.raises(FileNotFoundError):
        gif_to_images(
            osp.join(root, 'test.gif'), osp.join('123', 'img_folder'))
    gif_to_images(osp.join(root, 'test.gif'), osp.join(root, 'img_folder'))
    assert os.path.isdir(osp.join(root, 'img_folder'))
    assert images_to_array(osp.join(root,
                                    'img_folder')).shape[1:] == (512, 512, 3)

    # wrong inpath video_to_gif
    with pytest.raises(FileNotFoundError):
        video_to_gif(osp.join('123', 'test.mp4'), osp.join(root, 'test.gif'))
    # wrong outpath
    with pytest.raises(FileNotFoundError):
        video_to_gif(osp.join(root, 'test.mp4'), osp.join('123', 'test.gif'))
    video_to_gif(osp.join(root, 'test.mp4'), osp.join(root, 'test.gif'))
    assert os.path.isfile(osp.join(root, 'test.gif'))

    # wrong inpath video_to_images
    with pytest.raises(FileNotFoundError):
        video_to_images(
            osp.join('123', 'test.mp4'), osp.join(root, 'img_folder'))
    # wrong outpath
    with pytest.raises(FileNotFoundError):
        video_to_images(
            osp.join(root, 'test.mp4'), osp.join('123', 'img_folder'))
    video_to_images(osp.join(root, 'test.mp4'), osp.join(root, 'img_folder'))
    assert os.path.isdir(osp.join(root, 'img_folder'))
    assert images_to_array(osp.join(root,
                                    'img_folder')).shape[1:] == (512, 512, 3)


def test_concat_crop():
    # temporal_concat_video, spatial_concat_video,
    #  temporal_crop_video, spatial_crop_video
    v = np.random.randint(
        low=0, high=255, size=(30, 512, 512, 3), dtype=np.uint8)

    array_to_video(v, osp.join(root, 'test.mp4'))

    # wrong input/output
    with pytest.raises(FileNotFoundError):
        temporal_concat_video([
            osp.join(root, '123.mp4'),
            osp.join(root, 'test.mp4'),
            osp.join(root, 'test.mp4')
        ],
                              output_path=osp.join(root, 'test1.mp4'))
    with pytest.raises(FileNotFoundError):
        temporal_concat_video([
            osp.join(root, 'test.mp4'),
            osp.join(root, 'test.mp4'),
            osp.join(root, 'test.mp4')
        ],
                              output_path=osp.join('123', 'test1.mp4'))
    temporal_concat_video(
        [osp.join(root, 'test.mp4'),
         osp.join(root, 'test.gif')],
        output_path=osp.join(root, 'test1.mp4'))
    assert os.path.isfile(osp.join(root, 'test1.mp4'))

    # wrong input/output
    with pytest.raises(FileNotFoundError):
        spatial_concat_video([
            osp.join(root, '123.mp4'),
            osp.join(root, 'test.mp4'),
            osp.join(root, 'test.mp4')
        ],
                             output_path=osp.join(root, 'test2.mp4'),
                             array=[2, 2],
                             padding=1)
    with pytest.raises(FileNotFoundError):
        spatial_concat_video([
            osp.join(root, 'test.mp4'),
            osp.join(root, 'test.mp4'),
            osp.join(root, 'test.mp4')
        ],
                             output_path=osp.join('123', 'test2.mp4'),
                             array=[2, 2],
                             padding=1)
    spatial_concat_video([
        osp.join(root, 'test.mp4'),
        osp.join(root, 'test.mp4'),
        osp.join(root, 'test.mp4')
    ],
                         output_path=osp.join(root, 'test2.mp4'),
                         array=[2, 2],
                         padding=1)
    assert os.path.isfile(osp.join(root, 'test2.mp4'))

    # wrong input/output
    with pytest.raises(FileNotFoundError):
        temporal_crop_video(
            osp.join(root, '123.mp4'),
            osp.join(root, 'test3.mp4'),
            start=0,
            end=10)
    with pytest.raises(FileNotFoundError):
        temporal_crop_video(
            osp.join(root, 'test1.mp4'),
            osp.join('123', 'test3.mp4'),
            start=0,
            end=10)
    temporal_crop_video(
        osp.join(root, 'test1.mp4'),
        osp.join(root, 'test3.mp4'),
        start=0,
        end=10)

    # wrong box
    with pytest.raises(AssertionError):
        spatial_crop_video(
            osp.join(root, 'test1.mp4'),
            osp.join(root, 'test3.mp4'),
            box=[10, 10, -1, -1])
    # wrong input/output
    with pytest.raises(FileNotFoundError):
        spatial_crop_video(
            osp.join(root, '123.mp4'),
            osp.join(root, 'test3.mp4'),
            box=[10, 10, -1, -1])
    with pytest.raises(FileNotFoundError):
        spatial_crop_video(
            osp.join(root, '123.mp4'),
            osp.join('123', 'test3.mp4'),
            box=[10, 10, -1, -1])
    spatial_crop_video(
        osp.join(root, 'test1.mp4'),
        osp.join(root, 'test3.mp4'),
        box=[10, 10, 100, 100])
    assert os.path.isfile(osp.join(root, 'test3.mp4'))


def test_compress():
    compress_video(
        osp.join(root, 'test.mp4'),
        output_path=osp.join(root, 'test1.mp4'),
        compress_rate=1,
        down_sample_scale=1,
        fps=30)
    size1 = os.path.getsize(osp.join(root, 'test1.mp4'))

    compress_video(
        osp.join(root, 'test.mp4'),
        output_path=osp.join(root, 'test2.mp4'),
        compress_rate=2,
        down_sample_scale=1,
        fps=30)
    size2 = os.path.getsize(osp.join(root, 'test2.mp4'))

    compress_video(
        osp.join(root, 'test.mp4'),
        output_path=osp.join(root, 'test3.mp4'),
        compress_rate=3,
        down_sample_scale=1,
        fps=30)
    size3 = os.path.getsize(osp.join(root, 'test3.mp4'))

    compress_video(
        osp.join(root, 'test.mp4'),
        output_path=osp.join(root, 'test4.mp4'),
        compress_rate=3,
        down_sample_scale=1,
        fps=15)
    size4 = os.path.getsize(osp.join(root, 'test4.mp4'))

    compress_video(
        osp.join(root, 'test.mp4'),
        output_path=osp.join(root, 'test5.mp4'),
        compress_rate=3,
        down_sample_scale=2,
        fps=15)
    size5 = os.path.getsize(osp.join(root, 'test5.mp4'))

    assert size1 > size2 > size3 > size4 > size5
