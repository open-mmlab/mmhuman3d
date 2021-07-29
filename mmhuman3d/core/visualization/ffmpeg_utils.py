# -*- coding: utf-8 -*-
import os
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Tuple, Union

import ffmpeg
import numpy as np


def array_to_video(
    image_array: np.ndarray,
    output_path: str,
    fps: Union[int, float] = 30,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> NoReturn:
    """Convert an array to a video directly, gif not supported.

    Args:
        image_array (np.ndarray): shape should be (f * h * w * 3).
        output_path (str): output video file path.
        fps (Union[int, float, optional): fps. Defaults to 30.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                optional): (width, height) of the output video.
                Defaults to None.

    Raises:
        FileNotFoundError: check output path.
        TypeError: check input array.

    Returns:
        NoReturn.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError('Input should be np.ndarray.')
    assert image_array.ndim == 4
    assert image_array.shape[-1] == 3
    if not (Path(output_path).suffix.lower() in ['.mp4']
            and os.path.exists(Path(output_path).parent)):
        raise FileNotFoundError('Wrong output file format.')
    if resolution:
        width, height = resolution
    else:
        height, width = image_array.shape[1], image_array.shape[2]
    command = [
        'ffmpeg',
        '-y',  # (optional) overwrite output file if it exists
        '-f',
        'rawvideo',
        '-s',
        '%dx%d' % (width, height),  # size of one frame
        '-pix_fmt',
        'bgr24',
        '-r',
        str(fps),  # frames per second
        '-loglevel',
        'error',
        '-threads',
        '4',
        '-i',
        '-',  # The input comes from a pipe
        '-vcodec',
        'libx264',
        '-an',  # Tells FFMPEG not to expect any audio
        output_path,
    ]
    print(f'Running \"{" ".join(command)}\"')
    pipe = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    index = 0
    while True:
        index += 1
        if index >= image_array.shape[0]:
            break
        pipe.stdin.write(image_array[index].tobytes())

    pipe.stdin.close()
    pipe.stderr.close()
    pipe.wait()


def array_to_images(
    image_array: np.ndarray,
    output_folder: str,
    img_format: str = '%06d.png',
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> NoReturn:
    """Convert an array to images directly.

    Args:
        image_array (np.ndarray): shape should be (f * h * w * 3).
        output_folder (str): output folder for the images.
        img_format (str, optional): format of the images.
                Defaults to '%06d.png'.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                optional): resolution(width, height) of output.
                Defaults to None.

    Raises:
        FileNotFoundError: check output folder.
        TypeError: check input array.

    Returns:
        NoReturn
    """
    if not os.path.isdir(Path(output_folder).parent):
        raise FileNotFoundError('Wrong output path.')
    elif not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    if not isinstance(image_array, np.ndarray):
        raise TypeError('Input should be np.ndarray.')
    assert image_array.ndim == 4
    assert image_array.shape[-1] == 3
    if resolution:
        width, height = resolution
    else:
        height, width = image_array.shape[1], image_array.shape[2]
    command = [
        'ffmpeg',
        '-y',  # (optional) overwrite output file if it exists
        '-f',
        'rawvideo',
        '-s',
        '%dx%d' % (width, height),  # size of one frame
        '-pix_fmt',
        'bgr24',  # bgr24 for matching OpenCV
        '-loglevel',
        'error',
        '-threads',
        '4',
        '-i',
        '-',  # The input comes from a pipe
        '-f',
        'image2',
        os.path.join(output_folder, img_format),
    ]
    print(f'Running \"{" ".join(command)}\"')
    pipe = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
        close_fds=True)
    index = 0
    while True:
        if index >= image_array.shape[0]:
            break
        pipe.stdin.write(image_array[index].tobytes())
        index += 1
    pipe.stdin.close()
    pipe.stderr.close()
    pipe.wait()


def video_to_array(
    input_path: str,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> np.ndarray:
    """Read a video/gif as an array of (f * h * w * 3).
    Args:
        input_path (str): input path.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                optional): resolution(width, height) of output.
                Defaults to None.

    Raises:
        FileNotFoundError: check the input path.

    Returns:
        np.ndarray: shape will be (f * h * w * 3).
    """
    if not (os.path.isfile(input_path)
            and Path(input_path).suffix.lower() in ['.mp4', '.gif']):
        raise FileNotFoundError('Wrong input path.')
    info = vid_info(input_path)
    if resolution:
        width, height = resolution
    else:
        width, height = int(info['width']), int(info['height'])
    command = [
        'ffmpeg',
        '-i',
        input_path,
        '-pix_fmt',
        'bgr24',  # bgr24 for matching OpenCV
        '-s',
        '%dx%d' % (int(width), int(height)),
        '-f',
        'image2pipe',
        '-vcodec',
        'rawvideo',
        '-loglevel',
        'error',
        'pipe:'
    ]
    print(f'Running \"{" ".join(command)}\"')
    # Execute FFmpeg as sub-process with stdout as a pipe
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    # Read decoded video frames from the PIPE until no more frames to read
    array = []
    while True:
        # Read decoded video frame (in raw video format) from stdout process.
        buffer = process.stdout.read(width * height * 3)
        # Break the loop if buffer length is not W*H*3\
        # (when FFmpeg streaming ends).
        if len(buffer) != width * height * 3:
            break
        img = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
        array.append(img[np.newaxis])
    process.stdout.flush()
    process.stdout.close()
    process.wait()
    return np.concatenate(array)


def images_to_array(input_folder: str,
                    resolution: Optional[Union[Tuple[int, int],
                                               Tuple[float, float]]] = None,
                    img_format: str = '%06d.png') -> np.ndarray:
    """Read a folder of images as an array of (f * h * w * 3).

    Args:
        input_folder (str): folder of input images.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]]:
                resolution(width, height) of output. Defaults to None.
        img_format (str, optional): format of images to be read.
                Defaults to '%06d.png'.

    Raises:
        FileNotFoundError: check the input path.

    Returns:
        np.ndarray: shape will be (f * h * w * 3).
    """
    if not (os.path.isdir(input_folder)):
        raise FileNotFoundError('Wrong input folder.')
    info = vid_info(f'{input_folder}/{img_format}' % 1)
    width, height = int(info['width']), int(info['height'])
    if resolution:
        width, height = resolution
    else:
        width, height = int(info['width']), int(info['height'])
    command = [
        'ffmpeg',
        '-y',
        '-threads',
        '4',
        '-i',
        f'{input_folder}/{img_format}',
        "-f",
        "rawvideo",
        '-pix_fmt',
        'bgr24',  # bgr24 for matching OpenCV
        '-s',
        '%dx%d' % (int(width), int(height)),
        '-loglevel',
        'error',
        '-'
    ]
    print(f'Running \"{" ".join(command)}\"')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    # Read decoded video frames from the PIPE until no more frames to read
    array = []
    while True:
        # Read decoded video frame (in raw video format) from stdout process.
        buffer = process.stdout.read(width * height * 3)
        # Break the loop if buffer length is not W*H*3\
        # (when FFmpeg streaming ends).
        if len(buffer) != width * height * 3:
            break
        img = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
        array.append(img[np.newaxis])
    process.stdout.flush()
    process.stdout.close()
    process.wait()
    return np.concatenate(array)


class vid_info(object):

    def __init__(self, input_path) -> NoReturn:
        """Get video info with ffmpeg-python.

        Args:
            vid_file ([str]): video file path.

        Raises:
            FileNotFoundError: check the input path.

        Returns:
            NoReturn.
        """
        if not (os.path.isfile(input_path) and Path(input_path).suffix.lower()
                in ['.mp4', '.gif', '.png', '.jpg']):
            raise FileNotFoundError('Wrong input path.')
        probe = ffmpeg.probe(input_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print('No video stream found', file=sys.stderr)
            sys.exit(1)
        self.video_stream = video_stream

    def __getitem__(self, key: str):
        """
            Key ([str]): range in ['index', 'codec_name', 'codec_long_name',
            'profile', 'codec_type', 'codec_time_base', 'codec_tag_string',
            'codec_tag', 'width', 'height', 'coded_width', 'coded_height',
            'has_b_frames', 'pix_fmt', 'level', 'chroma_location', 'refs',
            'is_avc', 'nal_length_size', 'r_frame_rate', 'avg_frame_rate',
            'time_base', 'start_pts', 'start_time', 'duration_ts',
            'duration', 'bit_rate', 'bits_per_raw_sample', 'nb_frames',
            'disposition', 'tags']

        """
        return self.video_stream[key]


def video_to_gif(
    input_path: str,
    output_path: str,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> NoReturn:
    """Convert a video to a gif file.

    Args:
        input_path (str): video file path.
        output_path (str): gif file path.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                optional): (width, height) of the output video.
                Defaults to None.

    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        NoReturn.
    """
    if not (os.path.isfile(input_path)
            and Path(input_path).suffix.lower() in ['.mp4']):
        raise FileNotFoundError('Wrong input path.')
    if not (Path(output_path).suffix.lower() in ['.gif']
            and os.path.isdir(Path(output_path).parent)):
        raise FileNotFoundError('Wrong output path.')
    info = vid_info(input_path)
    if resolution:
        width, height = resolution
    else:
        width, height = int(info['width']), int(info['height'])

    command = [
        'ffmpeg', '-i', input_path, '-r', '15', '-s',
        '%dx%d' % (width, height), '-loglevel', 'error', '-threads', '4', '-y',
        output_path
    ]
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def video_to_images(
    input_path: str,
    output_folder: str,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> NoReturn:
    """Convert a video to a folder of images.

    Args:
        input_path (str): video file path
        output_folder (str): ouput folder to store the images
        resolution (Optional[Tuple[int, int]], optional):
                (width, height) of output. defaults to None.

    Raises:
        FileNotFoundError: check the input path
        FileNotFoundError: check the output path

    Returns:
        NoReturn
    """
    if not (os.path.isfile(input_path)
            and Path(input_path).suffix.lower() in ['.mp4', '.gif']):
        raise FileNotFoundError('Wrong input path.')
    if not os.path.isdir(Path(output_folder).parent):
        raise FileNotFoundError('Wrong output path.')
    elif not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    command = [
        'ffmpeg', '-i', input_path, '-f', 'image2', '-v', 'error', '-threads',
        '4', f'{output_folder}/%06d.png'
    ]
    if resolution:
        width, height = resolution
        command.insert(3, '-s')
        command.insert(4, '%dx%d' % (width, height))
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def images_to_video(
    input_folder: str,
    output_path: str,
    remove_raw_file: bool = False,
    img_format: str = '%06d.png',
    fps: Union[int, float] = 30,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> NoReturn:
    """Convert a folder of images to a video.

    Args:
        input_folder (str): input image folder
        output_path (str): output video file path
        remove_raw_file (bool, optional): whether remove raw images.
                    Defaults to False.
        img_format (str, optional): format to name the images].
                    Defaults to '%06d.png'.
        fps (Union[int, float], optional): output video fps. Defaults to 30.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                    optional): (width, height) of output.
                    defaults to None.

    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        NoReturn
    """
    if not (os.path.isdir(input_folder)):
        raise FileNotFoundError('Wrong input folder.')
    if not ((Path(output_path).suffix.lower() in ['.mp4'])
            and os.path.exists(Path(output_path).parent)):
        raise FileNotFoundError('Wrong output path.')
    command = [
        'ffmpeg',
        '-y',
        '-threads',
        '4',
        '-i',
        f'{input_folder}/{img_format}',
        '-profile:v',
        'baseline',
        '-level',
        '3.0',
        '-r',
        f'{fps}',
        '-c:v',
        'libx264',
        '-r',
        str(fps),
        '-pix_fmt',
        'yuv420p',
        '-an',
        '-v',
        'error',
        '-loglevel',
        'error',
        output_path,
    ]
    if resolution:
        width, height = resolution
        command.insert(1, '-s')
        command.insert(2, '%dx%d' % (width, height))
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    if remove_raw_file:
        shutil.rmtree(input_folder)


def images_to_gif(
    input_folder: str,
    output_path: str,
    remove_raw_file: bool = False,
    img_format: str = '%06d.png',
    fps: int = 15,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> NoReturn:
    """Convert series of images to a video, similar to images_to_video,
        but provide more suitable parameters.

    Args:
        input_folder (str): input image folder.
        output_path (str): output gif file path.
        remove_raw_file (bool, optional): whether remove raw images.
                Defaults to False.
        img_format (str, optional): format to name the images.
                Defaults to '%06d.png'.
        fps (int, optional): output video fps. Defaults to 15.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                optional): (width, height) of output. Defaults to None.

    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        NoReturn
    """
    if not (os.path.isdir(input_folder)):
        raise FileNotFoundError('Wrong input folder.')
    if not ((Path(output_path).suffix.lower() in ['.gif'])
            and os.path.exists(Path(output_path).parent)):
        raise FileNotFoundError('Wrong output path.')
    command = [
        'ffmpeg',
        '-y',
        '-threads',
        '4',
        '-i',
        f'{input_folder}/{img_format}',
        '-r',
        str(fps),
        '-loglevel',
        'error',
        '-v',
        'error',
        output_path,
    ]
    if resolution:
        width, height = resolution
        command.insert(1, '-s')
        command.insert(2, '%dx%d' % (width, height))
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    if remove_raw_file:
        shutil.rmtree(input_folder)


def gif_to_video(
    input_path: str,
    output_path: str,
    fps: int = 30,
    remove_raw_file: bool = False,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> NoReturn:
    """Convert a gif file to a video.

    Args:
        input_path (str): input gif file path.
        output_path (str): output video file path.
        fps (int, optional): fps. Defaults to 30.
        remove_raw_file (bool, optional): whether remove original input file.
                Defaults to False.
        down_sampl_scale (Union[int, float], optional): down sample scale.
                Defaults to 1.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                optional): (width, height) of output. Defaults to None.

    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        NoReturn
    """
    if not (os.path.isfile(input_path)
            and Path(input_path).suffix.lower() in ['.gif']):
        raise FileNotFoundError('Wrong input path.')
    if not ((Path(output_path).suffix.lower() in ['.mp4'])
            and os.path.exists(Path(output_path).parent)):
        raise FileNotFoundError('Wrong output path.')

    command = [
        'ffmpeg', '-i', input_path, '-r',
        str(fps), '-loglevel', 'error', '-y', output_path, '-threads', '4'
    ]
    if resolution:
        width, height = resolution
        command.insert(3, '-s')
        command.insert(4, '%dx%d' % (width, height))
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    if remove_raw_file:
        subprocess.call(['rm', '-f', input_path])


def gif_to_images(
    input_path: str,
    output_folder: str,
    fps: int = 30,
    img_format: str = '%06d.png',
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> NoReturn:
    """Convert a gif file to a folder of images.

    Args:
        input_path (str): input gif file path.
        output_folder (str): output folder to save the images.
        fps (int, optional): fps. Defaults to 30.
        img_format (str, optional): output image name format.
                Defaults to '%06d.png'.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                optional): (width, height) of output.
                Defaults to None.

    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        NoReturn
    """
    if not (os.path.isfile(input_path)
            and Path(input_path).suffix.lower() in ['.gif']):
        raise FileNotFoundError('Wrong input path.')

    if not os.path.isdir(Path(output_folder).parent):
        raise FileNotFoundError('Wrong output path.')
    elif not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    command = [
        'ffmpeg', '-i', input_path, '-r',
        str(fps), '-loglevel', 'error', '-f', 'image2', '-v', 'error',
        '-threads', '4', '-y', f'{output_folder}/{img_format}'
    ]
    if resolution:
        width, height = resolution
        command.insert(3, '-s')
        command.insert('%dx%d' % (width, height))
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def spatial_crop_video(
    input_path: str,
    output_path: str,
    box: Iterable[int] = [0, 0, 100, 100],
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> NoReturn:
    """Spatially crop a video or gif file.

    Args:
        input_path (str): input video or gif file path.
        output_path (str): output video or gif file path.
        box (Iterable[int], optional): [x, y of the crop region left.
                corner and width and height]. Defaults to [0, 0, 100, 100].
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                optional): (width, height) of output. Defaults to None.

    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        NoReturn
    """
    if not (os.path.isfile(input_path)
            and Path(input_path).suffix.lower() in ['.mp4', '.gif']):
        raise FileNotFoundError('Wrong input path.')
    if not os.path.isdir(Path(output_path).parent):
        raise FileNotFoundError('Wrong output path.')
    assert len(box) == 4
    x, y, w, h = box
    assert (w > 0 and h > 0)
    command = [
        'ffmpeg', '-i', input_path, '-vf',
        'crop=%d:%d:%d:%d' % (w, h, x, y), '-loglevel', 'error', '-y',
        output_path
    ]
    if resolution:
        width, height = resolution
        command.insert(3, '-s')
        command.insert(4, '%dx%d' % (width, height))
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def spatial_concat_video(input_path_list: List[str],
                         output_path: str,
                         array: List[int] = [1, 1],
                         direction='h',
                         resolution: Optional[Union[Tuple[int, int],
                                                    Tuple[float,
                                                          float]]] = (512,
                                                                      512),
                         remove_raw_files: bool = False,
                         padding: int = 0) -> NoReturn:
    """Spatially concat some videos as an array video.

    Args:
        input_path_list (list): input video or gif file list.
        output_path (str): output video or gif file path.
        array (List[int], optional): line number and column number of
                    the video array]. Defaults to [1, 1].
        direction (str, optional): [choose in 'h' or 'v', represent
                    horizontal and vertical separately].
                    Defaults to 'h'.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                    optional): (width, height) of output.
                    Defaults to (512, 512).
        remove_raw_files (bool, optional): whether remove raw images.
                    Defaults to False.
        padding (int, optional): width of pixels between videos.
                    Defaults to 0.

    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        NoReturn
    """
    lowercase = string.ascii_lowercase
    assert len(array) == 2
    assert (array[0] * array[1]) >= len(input_path_list)
    for path in input_path_list:
        if not (os.path.exists(path) and os.path.isfile(path)
                and Path(path).suffix.lower() in ['.mp4']):
            raise FileNotFoundError('Wrong input file path.')
    if not ((Path(output_path).suffix.lower() in ['.mp4'])
            and os.path.exists(Path(output_path).parent)):
        raise FileNotFoundError('Wrong output path.')
    command = ['ffmpeg']
    width, height = resolution
    scale_command = []
    for index, vid_file in enumerate(input_path_list):
        command.append('-i')
        command.append(vid_file)
        scale_command.append(
            '[%d:v]scale=%d:%d:force_original_aspect_ratio=0[v%d];' %
            (index, width, height, index))

    scale_command = ' '.join(scale_command)
    pad_command = "[v%d]pad=%d:%d[%s];" % (0, width * array[1] + padding *
                                           (array[1] - 1),
                                           height * array[0] + padding *
                                           (array[0] - 1), lowercase[0])
    for index in range(1, len(input_path_list)):
        if direction == 'h':
            pad_width = index % array[1] * (width + padding)
            pad_height = index // array[1] * (height + padding)
        else:
            pad_width = index % array[0] * (width + padding)
            pad_height = index // array[0] * (height + padding)

        pad_command += "[%s][v%d]overlay=%d:%d" % (lowercase[index - 1], index,
                                                   pad_width, pad_height)
        if index != len(input_path_list) - 1:
            pad_command += "[%s];" % lowercase[index]

    command += [
        '-filter_complex',
        "%s%s" % (scale_command, pad_command), '-loglevel', 'error', '-y',
        output_path
    ]
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    if remove_raw_files:
        command = ['rm', '-f'] + input_path_list
        subprocess.call(command)


def temporal_crop_video(
    input_path: str,
    output_path: str,
    start: int = 0,
    end: int = -1,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> NoReturn:
    """Temporally crop a video/gif into another video/gif.

    Args:
        input_path (str): input video or gif file path.
        output_path (str): output video of gif file path.
        start (int, optional): start frame index. Defaults to 0.
        end (int, optional): end frame index. Defaults to -1.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
                optional): (width, height) of output. Defaults to None.

    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        NoReturn
    """
    if not (os.path.isfile(input_path)
            and Path(input_path).suffix.lower() in ['.mp4', '.gif']):
        raise FileNotFoundError('Wrong input path.')
    if not ((Path(output_path).suffix.lower() in ['.mp4', '.gif'])
            and os.path.exists(Path(output_path).parent)):
        raise FileNotFoundError('Wrong output path.')
    info = vid_info(input_path)
    num_frames, time = int(info['nb_frames']), float(info['duration'])
    end = min(end, num_frames - 1)
    end = (num_frames + end) % num_frames
    fps = num_frames / float(time)
    start = start / fps
    end = end / fps
    command = [
        'ffmpeg', '-y', '-ss',
        str(start), '-t',
        str(end - start), '-accurate_seek', '-i', input_path, '-loglevel',
        'error', '-vcodec', 'libx264', output_path
    ]
    if resolution:
        width, height = resolution
        command.insert(1, '-s')
        command.insert(2, '%dx%d' % (width, height))
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def temporal_concat_video(input_path_list: List[str],
                          output_path: str,
                          resolution: Optional[Union[Tuple[int, int],
                                                     Tuple[float,
                                                           float]]] = (512,
                                                                       512),
                          remove_raw_files: bool = False) -> NoReturn:
    """Concat no matter videos or gifs into a temporal sequence,
        and save as a new video or gif file.

    Args:
        input_path_list (List[str]): list of input video paths.
        output_path (str): output video file path.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]]
                , optional): (width, height) of output]. Defaults to (512,512).
        remove_raw_files (bool, optional): whether remove the input videos.
                Defaults to False.

    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        NoReturn.
    """
    for path in input_path_list:
        if not (os.path.exists(path) and os.path.isfile(path)
                and Path(path).suffix.lower() in ['.mp4', '.gif']):
            raise FileNotFoundError('Wrong input file path.')
    if not ((Path(output_path).suffix.lower() in ['.mp4', '.gif'])
            and os.path.exists(Path(output_path).parent)):
        raise FileNotFoundError('Wrong output path.')
    width, height = resolution
    command = ['ffmpeg']
    concat_command = []
    scale_command = []
    for index, vid_file in enumerate(input_path_list):
        command.append('-i')
        command.append(vid_file)
        scale_command.append(
            '[%d:v]scale=%d:%d:force_original_aspect_ratio=0[v%d];' %
            (index, width, height, index))
        concat_command.append('[v%d]' % index)
    concat_command = ''.join(concat_command)
    scale_command = ''.join(scale_command)
    command += [
        '-filter_complex',
        "%s%sconcat=n=%d:v=1:a=0[v]" %
        (scale_command, concat_command, len(input_path_list)), '-loglevel',
        'error', '-map', '[v]', '-c:v', 'libx264', '-y', output_path
    ]
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    if remove_raw_files:
        command = ['rm'] + input_path_list
        subprocess.call(command)


def compress_video(input_path: str,
                   output_path: str,
                   compress_rate: int = 1,
                   down_sample_scale: Union[float, int] = 1,
                   fps: int = 30) -> NoReturn:
    """Compress a video file.

    Args:
        input_path (str): input video file path.
        output_path (str): output video file path.
        compress_rate (int, optional): compress rate, influents the bit rate.
                Defaults to 1.
        down_sample_scale (Union[float, int], optional): spatial down sample
                scale. Defaults to 1.
        fps (int, optional): [description]. Defaults to 30.

    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        NoReturn.
    """
    if not (os.path.isfile(input_path)
            and Path(input_path).suffix.lower() in ['.mp4', '.gif']):
        raise FileNotFoundError('Wrong input path.')
    if not ((Path(output_path).suffix.lower() in ['.mp4', '.gif'])
            and os.path.exists(Path(output_path).parent)):
        raise FileNotFoundError('Wrong output path.')
    info = vid_info(input_path)

    width = int(info['width'])
    height = int(info['height'])
    bit_rate = int(info['bit_rate'])
    duration = float(info['duration'])
    if (output_path == input_path) or (not output_path):
        temp_outpath = os.path.join(
            os.path.abspath(Path(input_path).parent),
            'temp_file' + Path(input_path).suffix)
    else:
        temp_outpath = output_path
    command = [
        'ffmpeg', '-y', '-i', input_path, '-loglevel', 'error', '-b:v',
        str(bit_rate / (compress_rate * down_sample_scale)), '-r',
        str(float(fps)), '-t',
        str(duration), '-s',
        '%dx%d' %
        (int(width / down_sample_scale), int(height / down_sample_scale)),
        temp_outpath
    ]
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    if (output_path == input_path) or (not output_path):
        subprocess.call(['mv', '-f', temp_outpath, input_path])
