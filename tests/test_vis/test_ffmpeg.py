import os
import os.path as osp

import numpy as np
import pytest

from mmhuman3d.utils.ffmpeg_utils import (
    array_to_images,
    array_to_video,
    compress_video,
    crop_video,
    gif_to_images,
    gif_to_video,
    images_to_array,
    images_to_gif,
    images_to_video,
    pad_for_libx264,
    slice_video,
    spatial_concat_video,
    temporal_concat_video,
    vid_info_reader,
    video_to_array,
    video_to_gif,
    video_to_images,
    video_writer,
)

root = 'tests/data/ffmpeg_test'


def test_writer():
    writer = video_writer(
        output_path=osp.join(root, 'demo.mp4'), resolution=(512, 512))
    video_array = np.ones(shape=[512, 512, 3])
    writer.write(image_array=video_array)


def test_pad():
    gray_image = np.ones(shape=[25, 45], dtype=np.uint8)
    pad_gray_image = pad_for_libx264(gray_image)
    assert pad_gray_image.shape[0] % 2 == 0
    assert pad_gray_image.shape[1] % 2 == 0

    rgb_image = np.ones(shape=[25, 45, 3], dtype=np.uint8)
    pad_rgb_image = pad_for_libx264(rgb_image)
    assert pad_rgb_image.shape[0] % 2 == 0
    assert pad_rgb_image.shape[1] % 2 == 0

    gray_array = np.ones(shape=[11, 25, 45], dtype=np.uint8)
    pad_gray_array = pad_for_libx264(gray_array)
    assert pad_gray_array.shape[1] % 2 == 0
    assert pad_gray_array.shape[2] % 2 == 0

    rgb_array = np.ones(shape=[11, 25, 45, 3], dtype=np.uint8)
    pad_rgb_array = pad_for_libx264(rgb_array)
    assert pad_rgb_array.shape[1] % 2 == 0
    assert pad_rgb_array.shape[2] % 2 == 0


def test_generate_data():
    os.makedirs(root, exist_ok=True)
    v = np.random.randint(
        low=0, high=255, size=(30, 512, 512, 3), dtype=np.uint8)
    array_to_images(v, osp.join(root, 'input_images'))
    array_to_video(v, osp.join(root, 'input_video.mp4'))
    images_to_gif(
        osp.join(root, 'input_images'), osp.join(root, 'input_gif.gif'))
    images_to_gif(
        osp.join(root, 'input_images'),
        osp.join(root, 'input_gif.gif'),
        img_format=None)


def test_array_saver():
    # not array
    with pytest.raises(TypeError):
        v = None
        array_to_video(v, osp.join(root, 'test_saver.mp4'))

    # wrong shape
    with pytest.raises(AssertionError):
        v = np.random.randint(
            low=0, high=255, size=(30, 512, 512, 4), dtype=np.uint8)
        array_to_video(v, osp.join(root, 'test_saver.mp4'))
        os.makedirs(osp.join(root, 'img_folder_saver'))
        array_to_images(v, osp.join(root, 'img_folder_saver'))
    with pytest.raises(AssertionError):
        v = np.random.randint(
            low=0, high=255, size=(512, 512, 3), dtype=np.uint8)
        array_to_video(v, osp.join(root, 'test_saver.mp4'))
        array_to_images(v, osp.join(root, 'img_folder_saver'))

    # GIF not supported
    with pytest.raises(FileNotFoundError):
        v = np.random.randint(
            low=0, high=255, size=(30, 512, 512, 3), dtype=np.uint8)
        array_to_video(v, osp.join(root, 'test_saver.gif'))

    v = np.random.randint(
        low=0, high=255, size=(30, 512, 512, 3), dtype=np.uint8)
    array_to_images(v, osp.join(root, 'img_folder_saver'))
    array_to_video(v, osp.join(root, 'test_saver.mp4'))
    array_to_video(v[:, :-1, :-1, :], osp.join(root, 'test_even.mp4'))
    assert os.path.isfile(osp.join(root, 'test_saver.mp4'))
    assert os.path.isfile(osp.join(root, 'img_folder_saver', '%06d.png' % 1))
    assert os.path.isfile(osp.join(root, 'test_even.mp4'))


def test_image_reader():
    # images_to_array, video_to_array, vid_info_reader
    with pytest.raises(FileNotFoundError):
        _ = images_to_array(osp.join('/NoSuchDir', 'img_folder_reader'))

    # shape should be (f, h, w, 3)
    v = images_to_array(osp.join(root, 'input_images'), resolution=(300, 200))
    assert v.shape[1:] == (300, 200, 3)
    v = images_to_array(
        osp.join(root, 'input_images'), resolution=(300, 200), img_format=None)
    v = video_to_array(
        osp.join(root, 'input_video.mp4'), resolution=(300, 200))
    assert v.shape[1:] == (300, 200, 3)

    vid = vid_info_reader(osp.join(root, 'input_video.mp4'))
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


def test_temporal_concat_video():
    # temporal_concat_video
    # wrong input/output
    with pytest.raises(FileNotFoundError):
        temporal_concat_video([
            osp.join(root, 'NoSuchFile_crop.mp4'),
            osp.join(root, 'input_video.mp4'),
            osp.join(root, 'input_video.mp4')
        ],
                              output_path=osp.join(
                                  root, 'test_temporal_concat_output.mp4'))
    v = np.random.randint(
        low=0, high=255, size=(30, 512, 512, 3), dtype=np.uint8)
    array_to_video(v, osp.join(root, 'input_video.mp4'))

    temporal_concat_video(
        [osp.join(root, 'input_video.mp4'),
         osp.join(root, 'input_video.mp4')],
        output_path=osp.join(root, 'test_temporal_concat_output.mp4'))
    assert os.path.isfile(osp.join(root, 'test_temporal_concat_output.mp4'))


def test_spatial_c1():
    # spatial_concat_video
    # wrong input/output
    with pytest.raises(FileNotFoundError):
        spatial_concat_video([
            osp.join(root, 'NoSuchFile.mp4'),
            osp.join(root, 'input_video.mp4'),
            osp.join(root, 'input_video.mp4')
        ],
                             output_path=osp.join(
                                 root, 'test_spacial_concat_output.mp4'),
                             array=[2, 2],
                             padding=1)

    spatial_concat_video([
        osp.join(root, 'input_video.mp4'),
        osp.join(root, 'input_video.mp4'),
        osp.join(root, 'input_video.mp4')
    ],
                         output_path=osp.join(
                             root, 'test_spacial_concat_output.mp4'),
                         array=[2, 2],
                         padding=1)
    assert os.path.isfile(osp.join(root, 'test_spacial_concat_output.mp4'))


def test_temporal_c2():
    # slice_video
    # wrong input/output
    with pytest.raises(FileNotFoundError):
        slice_video(
            osp.join(root, 'NoSuchFile.mp4'),
            osp.join(root, 'test_temporal_crop_output.mp4'),
            start=0,
            end=10)

    slice_video(
        osp.join(root, 'input_video.mp4'),
        osp.join(root, 'test_temporal_crop_output.mp4'),
        start=0,
        end=10)

    slice_video(
        osp.join(root, 'input_video.mp4'),
        osp.join(root, 'test_temporal_crop_output.mp4'),
        start=0,
        end=10,
        resolution=(50, 50))


def test_spacial_c2():
    # crop_video
    # wrong box
    with pytest.raises(AssertionError):
        crop_video(
            osp.join(root, 'input_video.mp4'),
            osp.join(root, 'test_spacial_crop_output.mp4'),
            box=[10, 10, -1, -1])
    # wrong input/output
    with pytest.raises(FileNotFoundError):
        crop_video(
            osp.join(root, 'NoSuchFile.mp4'),
            osp.join(root, 'test_spacial_crop_output.mp4'),
            box=[10, 10, -1, -1])

    crop_video(
        osp.join(root, 'input_video.mp4'),
        osp.join(root, 'test_spacial_crop_output.mp4'),
        box=[10, 10, 100, 100])
    assert os.path.isfile(osp.join(root, 'test_spacial_crop_output.mp4'))

    crop_video(
        osp.join(root, 'input_video.mp4'),
        osp.join(root, 'test_spacial_crop_output.mp4'),
        box=[10, 10, 100, 100],
        resolution=(40, 40))
    assert os.path.isfile(osp.join(root, 'test_spacial_crop_output.mp4'))


def test_convert():
    # images_to_gif, images_to_video, gif_to_video,
    # gif_to_images, video_to_gif, video_to_images

    # wrong inpath images_to_gif

    with pytest.raises(FileNotFoundError):
        images_to_gif(
            osp.join('/NoSuchDir', 'img_folder'),
            osp.join(root, 'images_to_gif_output.gif'))

    images_to_gif(
        osp.join(root, 'input_images'),
        osp.join(root, 'images_to_gif_output.gif'),
        fps=30)
    assert os.path.isfile(osp.join(root, 'images_to_gif_output.gif'))

    # wrong inpath images_to_video
    with pytest.raises(FileNotFoundError):
        images_to_video(
            osp.join('/NoSuchDir', 'input_images'),
            osp.join(root, 'images_to_video_output.mp4'))
    # wrong outpath
    images_to_video(
        osp.join(root, 'input_images'),
        osp.join(root, 'images_to_video_output.mp4'))
    assert os.path.isfile(osp.join(root, 'images_to_video_output.mp4'))
    images_to_video(
        osp.join(root, 'input_images'),
        osp.join(root, 'images_to_video_output.mp4'),
        img_format=None)
    assert os.path.isfile(osp.join(root, 'images_to_video_output.mp4'))
    images_to_video(
        osp.join(root, 'input_images'),
        osp.join(root, 'images_to_video_output.mp4'),
        resolution=(100, 100))
    assert os.path.isfile(osp.join(root, 'images_to_video_output.mp4'))

    # wrong inpath gif_to_video
    with pytest.raises(FileNotFoundError):
        gif_to_video(
            osp.join('/NoSuchDir', 'input_gif.gif'),
            osp.join(root, 'gif_to_video_output.mp4'))
    # wrong outpath
    gif_to_video(
        osp.join(root, 'input_gif.gif'),
        osp.join(root, 'gif_to_video_output.mp4'))
    assert os.path.isfile(osp.join(root, 'gif_to_video_output.mp4'))

    # wrong inpath gif_to_images
    with pytest.raises(FileNotFoundError):
        gif_to_images(
            osp.join('/NoSuchDir', 'input_gif.gif'),
            osp.join(root, 'gif_to_images_output'))
    gif_to_images(
        osp.join(root, 'input_gif.gif'), osp.join(root,
                                                  'gif_to_images_output'))
    assert os.path.isdir(osp.join(root, 'gif_to_images_output'))
    assert images_to_array(osp.join(
        root, 'gif_to_images_output')).shape[1:] == (512, 512, 3)

    # wrong inpath video_to_gif
    with pytest.raises(FileNotFoundError):
        video_to_gif(
            osp.join('/NoSuchDir', 'input_video.mp4'),
            osp.join(root, 'video_to_gif_output.gif'))
    video_to_gif(
        osp.join(root, 'input_video.mp4'),
        osp.join(root, 'video_to_gif_output.gif'))
    assert os.path.isfile(osp.join(root, 'video_to_gif_output.gif'))

    # wrong inpath video_to_images
    with pytest.raises(FileNotFoundError):
        video_to_images(
            osp.join('/NoSuchDir', 'input_video.mp4'),
            osp.join(root, 'video_to_images_output'))

    video_to_images(
        osp.join(root, 'input_video.mp4'),
        osp.join(root, 'video_to_images_output'))
    assert os.path.isdir(osp.join(root, 'video_to_images_output'))
    assert images_to_array(osp.join(
        root, 'video_to_images_output')).shape[1:] == (512, 512, 3)


def test_compress():
    compress_video(
        osp.join(root, 'input_video.mp4'),
        output_path=osp.join(root, 'compress_video_output_rate1.mp4'),
        compress_rate=1,
        down_sample_scale=1,
        fps=30)
    size1 = os.path.getsize(osp.join(root, 'compress_video_output_rate1.mp4'))

    compress_video(
        osp.join(root, 'input_video.mp4'),
        output_path=osp.join(root, 'compress_video_output_rate2.mp4'),
        compress_rate=2,
        down_sample_scale=1,
        fps=30)
    size2 = os.path.getsize(osp.join(root, 'compress_video_output_rate2.mp4'))

    compress_video(
        osp.join(root, 'input_video.mp4'),
        output_path=osp.join(root, 'compress_video_output_rate3.mp4'),
        compress_rate=3,
        down_sample_scale=1,
        fps=30)
    size3 = os.path.getsize(osp.join(root, 'compress_video_output_rate3.mp4'))

    compress_video(
        osp.join(root, 'input_video.mp4'),
        output_path=osp.join(root, 'compress_video_output_rate4.mp4'),
        compress_rate=3,
        down_sample_scale=2,
        fps=15)
    size4 = os.path.getsize(osp.join(root, 'compress_video_output_rate4.mp4'))

    assert size1 > size2
    assert size2 > size3
    assert size3 > size4
