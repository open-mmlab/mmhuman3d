import glob
import json
import os
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from mmhuman3d.utils.path_utils import check_input_path, prepare_output_path

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class video_writer:

    def __init__(self,
                 output_path: str,
                 resolution: Iterable[int],
                 fps: float = 30.0,
                 num_frame: int = 1e9,
                 disable_log: bool = False) -> None:
        prepare_output_path(
            output_path,
            allowed_suffix=['.mp4'],
            tag='output video',
            path_type='file',
            overwrite=True)
        height, width = resolution
        width += width % 2
        height += height % 2
        command = [
            'ffmpeg',
            '-y',  # (optional) overwrite output file if it exists
            '-f',
            'rawvideo',
            '-pix_fmt',
            'bgr24',
            '-s',
            f'{int(width)}x{int(height)}',
            '-r',
            f'{fps}',  # frames per second
            '-loglevel',
            'error',
            '-threads',
            '1',
            '-i',
            '-',  # The input comes from a pipe
            '-vcodec',
            'libx264',
            '-r',
            f'{fps}',  # frames per second
            '-an',  # Tells FFMPEG not to expect any audio
            output_path,
        ]
        if not disable_log:
            print(f'Running \"{" ".join(command)}\"')
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None or process.stderr is None:
            raise BrokenPipeError('No buffer received.')
        self.process = process
        self.num_frame = num_frame
        self.len = 0

    def write(self, image_array: np.ndarray):
        if self.len <= self.num_frame:
            try:
                self.process.stdin.write(image_array.tobytes())
                self.len += 1
            except KeyboardInterrupt:
                self.__del__()

    def __del__(self):
        self.process.stdin.close()
        self.process.stderr.close()
        self.process.wait()


def array_to_video(
    image_array: np.ndarray,
    output_path: str,
    fps: Union[int, float] = 30,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
    disable_log: bool = False,
) -> None:
    """Convert an array to a video directly, gif not supported.

    Args:
        image_array (np.ndarray): shape should be (f * h * w * 3).
        output_path (str): output video file path.
        fps (Union[int, float, optional): fps. Defaults to 30.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
            optional): (height, width) of the output video.
            Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check output path.
        TypeError: check input array.

    Returns:
        None.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError('Input should be np.ndarray.')
    assert image_array.ndim == 4
    assert image_array.shape[-1] == 3
    prepare_output_path(
        output_path,
        allowed_suffix=['.mp4'],
        tag='output video',
        path_type='file',
        overwrite=True)
    if resolution:
        height, width = resolution
        width += width % 2
        height += height % 2
    else:
        image_array = pad_for_libx264(image_array)
        height, width = image_array.shape[1], image_array.shape[2]
    command = [
        'ffmpeg',
        '-y',  # (optional) overwrite output file if it exists
        '-f',
        'rawvideo',
        '-s',
        f'{int(width)}x{int(height)}',  # size of one frame
        '-pix_fmt',
        'bgr24',
        '-r',
        f'{fps}',  # frames per second
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
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        raise BrokenPipeError('No buffer received.')
    index = 0
    while True:
        if index >= image_array.shape[0]:
            break
        process.stdin.write(image_array[index].tobytes())
        index += 1
    process.stdin.close()
    process.stderr.close()
    process.wait()


def array_to_images(
    image_array: np.ndarray,
    output_folder: str,
    img_format: str = '%06d.png',
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
    disable_log: bool = False,
) -> None:
    """Convert an array to images directly.

    Args:
        image_array (np.ndarray): shape should be (f * h * w * 3).
        output_folder (str): output folder for the images.
        img_format (str, optional): format of the images.
            Defaults to '%06d.png'.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
            optional): resolution(height, width) of output.
            Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.

    Raises:
        FileNotFoundError: check output folder.
        TypeError: check input array.

    Returns:
        None
    """
    prepare_output_path(
        output_folder,
        allowed_suffix=[],
        tag='output image folder',
        path_type='dir',
        overwrite=True)

    if not isinstance(image_array, np.ndarray):
        raise TypeError('Input should be np.ndarray.')
    assert image_array.ndim == 4
    assert image_array.shape[-1] == 3
    if resolution:
        height, width = resolution
    else:
        height, width = image_array.shape[1], image_array.shape[2]
    command = [
        'ffmpeg',
        '-y',  # (optional) overwrite output file if it exists
        '-f',
        'rawvideo',
        '-s',
        f'{int(width)}x{int(height)}',  # size of one frame
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
        '-start_number',
        '0',
        os.path.join(output_folder, img_format),
    ]
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
        close_fds=True)
    if process.stdin is None or process.stderr is None:
        raise BrokenPipeError('No buffer received.')
    index = 0
    while True:
        if index >= image_array.shape[0]:
            break
        process.stdin.write(image_array[index].tobytes())
        index += 1
    process.stdin.close()
    process.stderr.close()
    process.wait()


def video_to_array(
    input_path: str,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
    start: int = 0,
    end: Optional[int] = None,
    disable_log: bool = False,
) -> np.ndarray:
    """
    Read a video/gif as an array of (f * h * w * 3).

    Args:
        input_path (str): input path.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
            optional): resolution(height, width) of output.
            Defaults to None.
        start (int, optional): start frame index. Inclusive.
             If < 0, will be converted to frame_index range in [0, frame_num].
            Defaults to 0.
        end (int, optional): end frame index. Exclusive.
            Could be positive int or negative int or None.
            If None, all frames from start till the last frame are included.
            Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.

    Raises:
        FileNotFoundError: check the input path.

    Returns:
        np.ndarray: shape will be (f * h * w * 3).
    """
    check_input_path(
        input_path,
        allowed_suffix=['.mp4', 'mkv', 'avi', '.gif'],
        tag='input video',
        path_type='file')

    info = vid_info_reader(input_path)
    if resolution:
        height, width = resolution
    else:
        width, height = int(info['width']), int(info['height'])
    num_frames = int(info['nb_frames'])
    start = (min(start, num_frames - 1) + num_frames) % num_frames
    end = (min(end, num_frames - 1) +
           num_frames) % num_frames if end is not None else num_frames
    command = [
        'ffmpeg',
        '-i',
        input_path,
        '-filter_complex',
        f'[0]trim=start_frame={start}:end_frame={end}[v0]',
        '-map',
        '[v0]',
        '-pix_fmt',
        'bgr24',  # bgr24 for matching OpenCV
        '-s',
        f'{int(width)}x{int(height)}',
        '-f',
        'image2pipe',
        '-vcodec',
        'rawvideo',
        '-loglevel',
        'error',
        'pipe:'
    ]
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    # Execute FFmpeg as sub-process with stdout as a pipe
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    if process.stdout is None:
        raise BrokenPipeError('No buffer received.')
    # Read decoded video frames from the PIPE until no more frames to read
    array = []
    while True:
        # Read decoded video frame (in raw video format) from stdout process.
        buffer = process.stdout.read(int(width * height * 3))
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


def images_to_sorted_images(input_folder, output_folder, img_format='%06d'):
    """Copy and rename a folder of images into a new folder following the
    `img_format`.

    Args:
        input_folder (str): input folder.
        output_folder (str): output folder.
        img_format (str, optional): image format name, do not need extension.
            Defaults to '%06d'.

    Returns:
        str: image format of the rename images.
    """
    img_format = img_format.rsplit('.', 1)[0]
    file_list = []
    os.makedirs(output_folder, exist_ok=True)
    pngs = glob.glob(os.path.join(input_folder, '*.png'))
    if pngs:
        ext = 'png'
    file_list.extend(pngs)
    jpgs = glob.glob(os.path.join(input_folder, '*.jpg'))
    if jpgs:
        ext = 'jpg'
    file_list.extend(jpgs)
    file_list.sort()
    for index, file_name in enumerate(file_list):
        shutil.copy(
            file_name,
            os.path.join(output_folder, (img_format + '.%s') % (index, ext)))
    return img_format + '.%s' % ext


def images_to_array(
    input_folder: str,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
    img_format: str = '%06d.png',
    start: int = 0,
    end: Optional[int] = None,
    remove_raw_files: bool = False,
    disable_log: bool = False,
) -> np.ndarray:
    """
    Read a folder of images as an array of (f * h * w * 3).

    Args:
        input_folder (str): folder of input images.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]]:
            resolution(height, width) of output. Defaults to None.
        img_format (str, optional): format of images to be read.
            Defaults to '%06d.png'.
        start (int, optional): start frame index. Inclusive.
             If < 0, will be converted to frame_index range in [0, frame_num].
            Defaults to 0.
        end (int, optional): end frame index. Exclusive.
            Could be positive int or negative int or None.
            If None, all frames from start till the last frame are included.
            Defaults to None.
        remove_raw_files (bool, optional): whether remove raw images.
            Defaults to False.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path.

    Returns:
        np.ndarray: shape will be (f * h * w * 3).
    """
    check_input_path(
        input_folder,
        allowed_suffix=[''],
        tag='input image folder',
        path_type='dir')

    input_folderinfo = Path(input_folder)

    temp_input_folder = None
    if img_format is None:
        temp_input_folder = os.path.join(input_folderinfo.parent,
                                         input_folderinfo.name + '_temp')
        img_format = images_to_sorted_images(
            input_folder=input_folder, output_folder=temp_input_folder)
        input_folder = temp_input_folder

    info = vid_info_reader(f'{input_folder}/{img_format}' % start)
    width, height = int(info['width']), int(info['height'])
    if resolution:
        height, width = resolution
    else:
        width, height = int(info['width']), int(info['height'])

    num_frames = len(os.listdir(input_folder))
    start = (min(start, num_frames - 1) + num_frames) % num_frames
    end = (min(end, num_frames - 1) +
           num_frames) % num_frames if end is not None else num_frames
    command = [
        'ffmpeg',
        '-y',
        '-threads',
        '1',
        '-start_number',
        f'{start}',
        '-i',
        f'{input_folder}/{img_format}',
        '-frames:v',
        f'{end - start}',
        '-f',
        'rawvideo',
        '-pix_fmt',
        'bgr24',  # bgr24 for matching OpenCV
        '-s',
        f'{int(width)}x{int(height)}',
        '-loglevel',
        'error',
        '-'
    ]
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    if process.stdout is None:
        raise BrokenPipeError('No buffer received.')
    # Read decoded video frames from the PIPE until no more frames to read
    array = []
    while True:
        # Read decoded video frame (in raw video format) from stdout process.
        buffer = process.stdout.read(int(width * height * 3))
        # Break the loop if buffer length is not W*H*3\
        # (when FFmpeg streaming ends).

        if len(buffer) != width * height * 3:
            break
        img = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
        array.append(img[np.newaxis])
    process.stdout.flush()
    process.stdout.close()
    process.wait()
    if temp_input_folder is not None:
        if Path(temp_input_folder).is_dir():
            shutil.rmtree(temp_input_folder)
    if remove_raw_files:
        if Path(input_folder).is_dir():
            shutil.rmtree(input_folder)

    return np.concatenate(array)


class vid_info_reader(object):

    def __init__(self, input_path) -> None:
        """Get video information from video, mimiced from ffmpeg-python.
        https://github.com/kkroening/ffmpeg-python.

        Args:
            vid_file ([str]): video file path.

        Raises:
            FileNotFoundError: check the input path.

        Returns:
            None.
        """
        check_input_path(
            input_path,
            allowed_suffix=['.mp4', '.gif', '.png', '.jpg', '.jpeg'],
            tag='input file',
            path_type='file')
        cmd = [
            'ffprobe', '-show_format', '-show_streams', '-of', 'json',
            input_path
        ]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate()
        probe = json.loads(out.decode('utf-8'))
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print('No video stream found', file=sys.stderr)
            sys.exit(1)
        self.video_stream = video_stream

    def __getitem__(
        self,
        key: Literal['index', 'codec_name', 'codec_long_name', 'profile',
                     'codec_type', 'codec_time_base', 'codec_tag_string',
                     'codec_tag', 'width', 'height', 'coded_width',
                     'coded_height', 'has_b_frames', 'pix_fmt', 'level',
                     'chroma_location', 'refs', 'is_avc', 'nal_length_size',
                     'r_frame_rate', 'avg_frame_rate', 'time_base',
                     'start_pts', 'start_time', 'duration_ts', 'duration',
                     'bit_rate', 'bits_per_raw_sample', 'nb_frames',
                     'disposition', 'tags']):
        """Key (str): select in ['index', 'codec_name', 'codec_long_name',
        'profile', 'codec_type', 'codec_time_base', 'codec_tag_string',
        'codec_tag', 'width', 'height', 'coded_width', 'coded_height',
        'has_b_frames', 'pix_fmt', 'level', 'chroma_location', 'refs',
        'is_avc', 'nal_length_size', 'r_frame_rate', 'avg_frame_rate',
        'time_base', 'start_pts', 'start_time', 'duration_ts', 'duration',
        'bit_rate', 'bits_per_raw_sample', 'nb_frames', 'disposition',
        'tags']"""
        return self.video_stream[key]


def video_to_gif(
    input_path: str,
    output_path: str,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
    fps: Union[float, int] = 15,
    disable_log: bool = False,
) -> None:
    """Convert a video to a gif file.

    Args:
        input_path (str): video file path.
        output_path (str): gif file path.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
            optional): (height, width) of the output video.
            Defaults to None.
        fps (Union[float, int], optional): frames per second. Defaults to 15.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.

    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        None.
    """
    check_input_path(
        input_path,
        allowed_suffix=['.mp4'],
        tag='input video',
        path_type='file')
    prepare_output_path(
        output_path,
        allowed_suffix=['.gif'],
        tag='output gif',
        path_type='file',
        overwrite=True)

    info = vid_info_reader(input_path)
    duration = info['duration']
    if resolution:
        height, width = resolution
    else:
        width, height = int(info['width']), int(info['height'])

    command = [
        'ffmpeg', '-r',
        str(info['r_frame_rate']), '-i', input_path, '-r', f'{fps}', '-s',
        f'{width}x{height}', '-loglevel', 'error', '-t', f'{duration}',
        '-threads', '4', '-y', output_path
    ]
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def video_to_images(input_path: str,
                    output_folder: str,
                    resolution: Optional[Union[Tuple[int, int],
                                               Tuple[float, float]]] = None,
                    img_format: str = '%06d.png',
                    start: int = 0,
                    end: Optional[int] = None,
                    disable_log: bool = False) -> None:
    """Convert a video to a folder of images.

    Args:
        input_path (str): video file path
        output_folder (str): output folder to store the images
        resolution (Optional[Tuple[int, int]], optional):
            (height, width) of output. defaults to None.
        img_format (str, optional): format of images to be read.
            Defaults to '%06d.png'.
        start (int, optional): start frame index. Inclusive.
             If < 0, will be converted to frame_index range in [0, frame_num].
            Defaults to 0.
        end (int, optional): end frame index. Exclusive.
            Could be positive int or negative int or None.
            If None, all frames from start till the last frame are included.
            Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path
        FileNotFoundError: check the output path

    Returns:
        None
    """
    check_input_path(
        input_path,
        allowed_suffix=['.mp4'],
        tag='input video',
        path_type='file')
    prepare_output_path(
        output_folder,
        allowed_suffix=[],
        tag='output image folder',
        path_type='dir',
        overwrite=True)
    info = vid_info_reader(input_path)
    num_frames = int(info['nb_frames'])
    start = (min(start, num_frames - 1) + num_frames) % num_frames
    end = (min(end, num_frames - 1) +
           num_frames) % num_frames if end is not None else num_frames

    command = [
        'ffmpeg', '-i', input_path, '-filter_complex',
        f'[0]trim=start_frame={start}:end_frame={end}[v0]', '-map', '[v0]',
        '-f', 'image2', '-v', 'error', '-start_number', '0', '-threads', '1',
        f'{output_folder}/{img_format}'
    ]
    if resolution:
        height, width = resolution
        command.insert(3, '-s')
        command.insert(4, '%dx%d' % (width, height))
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def images_to_video(input_folder: str,
                    output_path: str,
                    remove_raw_file: bool = False,
                    img_format: str = '%06d.png',
                    fps: Union[int, float] = 30,
                    resolution: Optional[Union[Tuple[int, int],
                                               Tuple[float, float]]] = None,
                    start: int = 0,
                    end: Optional[int] = None,
                    disable_log: bool = False) -> None:
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
            optional): (height, width) of output.
            defaults to None.
        start (int, optional): start frame index. Inclusive.
            If < 0, will be converted to frame_index range in [0, frame_num].
            Defaults to 0.
        end (int, optional): end frame index. Exclusive.
            Could be positive int or negative int or None.
            If None, all frames from start till the last frame are included.
            Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        None
    """
    check_input_path(
        input_folder,
        allowed_suffix=[],
        tag='input image folder',
        path_type='dir')
    prepare_output_path(
        output_path,
        allowed_suffix=['.mp4'],
        tag='output video',
        path_type='file',
        overwrite=True)
    input_folderinfo = Path(input_folder)
    num_frames = len(os.listdir(input_folder))
    start = (min(start, num_frames - 1) + num_frames) % num_frames
    end = (min(end, num_frames - 1) +
           num_frames) % num_frames if end is not None else num_frames
    temp_input_folder = None
    if img_format is None:
        temp_input_folder = os.path.join(input_folderinfo.parent,
                                         input_folderinfo.name + '_temp')
        img_format = images_to_sorted_images(input_folder, temp_input_folder)

    command = [
        'ffmpeg',
        '-y',
        '-threads',
        '4',
        '-start_number',
        f'{start}',
        '-r',
        f'{fps}',
        '-i',
        f'{input_folder}/{img_format}'
        if temp_input_folder is None else f'{temp_input_folder}/{img_format}',
        '-frames:v',
        f'{end - start}',
        '-profile:v',
        'baseline',
        '-level',
        '3.0',
        '-c:v',
        'libx264',
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
        height, width = resolution
        width += width % 2
        height += height % 2
        command.insert(1, '-s')
        command.insert(2, '%dx%d' % (width, height))
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    if remove_raw_file:
        if Path(input_folder).is_dir():
            shutil.rmtree(input_folder)
    if temp_input_folder is not None:
        if Path(temp_input_folder).is_dir():
            shutil.rmtree(temp_input_folder)


def images_to_gif(
    input_folder: str,
    output_path: str,
    remove_raw_file: bool = False,
    img_format: str = '%06d.png',
    fps: int = 15,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
    start: int = 0,
    end: Optional[int] = None,
    disable_log: bool = False,
) -> None:
    """Convert series of images to a video, similar to images_to_video, but
    provide more suitable parameters.

    Args:
        input_folder (str): input image folder.
        output_path (str): output gif file path.
        remove_raw_file (bool, optional): whether remove raw images.
            Defaults to False.
        img_format (str, optional): format to name the images.
            Defaults to '%06d.png'.
        fps (int, optional): output video fps. Defaults to 15.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
            optional): (height, width) of output. Defaults to None.
        start (int, optional): start frame index. Inclusive.
            If < 0, will be converted to frame_index range in [0, frame_num].
            Defaults to 0.
        end (int, optional): end frame index. Exclusive.
            Could be positive int or negative int or None.
            If None, all frames from start till the last frame are included.
            Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        None
    """
    input_folderinfo = Path(input_folder)
    check_input_path(
        input_folder,
        allowed_suffix=[],
        tag='input image folder',
        path_type='dir')
    prepare_output_path(
        output_path,
        allowed_suffix=['.gif'],
        tag='output gif',
        path_type='file',
        overwrite=True)
    num_frames = len(os.listdir(input_folder))
    start = (min(start, num_frames - 1) + num_frames) % num_frames
    end = (min(end, num_frames - 1) +
           num_frames) % num_frames if end is not None else num_frames
    temp_input_folder = None
    if img_format is None:
        file_list = []
        temp_input_folder = os.path.join(input_folderinfo.parent,
                                         input_folderinfo.name + '_temp')
        os.makedirs(temp_input_folder, exist_ok=True)
        pngs = glob.glob(os.path.join(input_folder, '*.png'))
        ext = 'png'
        if pngs:
            ext = 'png'
        file_list.extend(pngs)
        jpgs = glob.glob(os.path.join(input_folder, '*.jpg'))
        if jpgs:
            ext = 'jpg'
        file_list.extend(jpgs)
        file_list.sort()
        for index, file_name in enumerate(file_list):
            shutil.copy(
                file_name,
                os.path.join(temp_input_folder, '%06d.%s' % (index + 1, ext)))
        input_folder = temp_input_folder
        img_format = '%06d.' + ext

    command = [
        'ffmpeg',
        '-y',
        '-threads',
        '4',
        '-start_number',
        f'{start}',
        '-r',
        f'{fps}',
        '-i',
        f'{input_folder}/{img_format}',
        '-frames:v',
        f'{end - start}',
        '-loglevel',
        'error',
        '-v',
        'error',
        output_path,
    ]
    if resolution:
        height, width = resolution
        command.insert(1, '-s')
        command.insert(2, '%dx%d' % (width, height))
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    if remove_raw_file:
        shutil.rmtree(input_folder)
    if temp_input_folder is not None:
        shutil.rmtree(temp_input_folder)


def gif_to_video(input_path: str,
                 output_path: str,
                 fps: int = 30,
                 remove_raw_file: bool = False,
                 resolution: Optional[Union[Tuple[int, int],
                                            Tuple[float, float]]] = None,
                 disable_log: bool = False) -> None:
    """Convert a gif file to a video.

    Args:
        input_path (str): input gif file path.
        output_path (str): output video file path.
        fps (int, optional): fps. Defaults to 30.
        remove_raw_file (bool, optional): whether remove original input file.
            Defaults to False.
        down_sample_scale (Union[int, float], optional): down sample scale.
            Defaults to 1.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
            optional): (height, width) of output. Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        None
    """
    check_input_path(
        input_path, allowed_suffix=['.gif'], tag='input gif', path_type='file')
    prepare_output_path(
        output_path,
        allowed_suffix=['.mp4'],
        tag='output video',
        path_type='file',
        overwrite=True)
    command = [
        'ffmpeg', '-i', input_path, '-r', f'{fps}', '-loglevel', 'error', '-y',
        output_path, '-threads', '4'
    ]
    if resolution:
        height, width = resolution
        command.insert(3, '-s')
        command.insert(4, '%dx%d' % (width, height))
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    if remove_raw_file:
        subprocess.call(['rm', '-f', input_path])


def gif_to_images(input_path: str,
                  output_folder: str,
                  fps: int = 30,
                  img_format: str = '%06d.png',
                  resolution: Optional[Union[Tuple[int, int],
                                             Tuple[float, float]]] = None,
                  disable_log: bool = False) -> None:
    """Convert a gif file to a folder of images.

    Args:
        input_path (str): input gif file path.
        output_folder (str): output folder to save the images.
        fps (int, optional): fps. Defaults to 30.
        img_format (str, optional): output image name format.
            Defaults to '%06d.png'.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
            optional): (height, width) of output.
            Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        None
    """
    check_input_path(
        input_path, allowed_suffix=['.gif'], tag='input gif', path_type='file')
    prepare_output_path(
        output_folder,
        allowed_suffix=[],
        tag='output image folder',
        path_type='dir',
        overwrite=True)
    command = [
        'ffmpeg', '-r', f'{fps}', '-i', input_path, '-loglevel', 'error', '-f',
        'image2', '-v', 'error', '-threads', '4', '-y', '-start_number', '0',
        f'{output_folder}/{img_format}'
    ]
    if resolution:
        height, width = resolution
        command.insert(3, '-s')
        command.insert(4, '%dx%d' % (width, height))
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def crop_video(
    input_path: str,
    output_path: str,
    box: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
    disable_log: bool = False,
) -> None:
    """Spatially or temporally crop a video or gif file.

    Args:
        input_path (str): input video or gif file path.
        output_path (str): output video or gif file path.
        box (Iterable[int], optional): [x, y of the crop region left.
            corner and width and height]. Defaults to [0, 0, 100, 100].
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
            optional): (height, width) of output. Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        None'-start_number', f'{start}',
    """
    check_input_path(
        input_path,
        allowed_suffix=['.gif', '.mp4'],
        tag='input video',
        path_type='file')
    prepare_output_path(
        output_path,
        allowed_suffix=['.gif', '.mp4'],
        tag='output video',
        path_type='file',
        overwrite=True)

    info = vid_info_reader(input_path)
    width, height = int(info['width']), int(info['height'])

    if box is None:
        box = [0, 0, width, height]

    assert len(box) == 4
    x, y, w, h = box
    assert (w > 0 and h > 0)
    command = [
        'ffmpeg', '-i', input_path, '-vcodec', 'libx264', '-vf',
        'crop=%d:%d:%d:%d' % (w, h, x, y), '-loglevel', 'error', '-y',
        output_path
    ]
    if resolution:
        height, width = resolution
        width += width % 2
        height += height % 2
        command.insert(-1, '-s')
        command.insert(-1, '%dx%d' % (width, height))
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def slice_video(input_path: str,
                output_path: str,
                start: int = 0,
                end: Optional[int] = None,
                resolution: Optional[Union[Tuple[int, int],
                                           Tuple[float, float]]] = None,
                disable_log: bool = False) -> None:
    """Temporally crop a video/gif into another video/gif.

    Args:
        input_path (str): input video or gif file path.
        output_path (str): output video of gif file path.
        start (int, optional): start frame index. Defaults to 0.
        end (int, optional): end frame index. Exclusive.
            Could be positive int or negative int or None.
            If None, all frames from start till the last frame are included.
            Defaults to None.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
            optional): (height, width) of output. Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        NoReturn
    """
    info = vid_info_reader(input_path)
    num_frames = int(info['nb_frames'])
    start = (min(start, num_frames - 1) + num_frames) % num_frames
    end = (min(end, num_frames - 1) +
           num_frames) % num_frames if end is not None else num_frames
    command = [
        'ffmpeg', '-y', '-i', input_path, '-filter_complex',
        f'[0]trim=start_frame={start}:end_frame={end}[v0]', '-map', '[v0]',
        '-loglevel', 'error', '-vcodec', 'libx264', output_path
    ]
    if resolution:
        height, width = resolution
        width += width % 2
        height += height % 2
        command.insert(1, '-s')
        command.insert(2, '%dx%d' % (width, height))
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def spatial_concat_video(input_path_list: List[str],
                         output_path: str,
                         array: List[int] = [1, 1],
                         direction: Literal['h', 'w'] = 'h',
                         resolution: Union[Tuple[int,
                                                 int], List[int], List[float],
                                           Tuple[float, float]] = (512, 512),
                         remove_raw_files: bool = False,
                         padding: int = 0,
                         disable_log: bool = False) -> None:
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
            optional): (height, width) of output.
            Defaults to (512, 512).
        remove_raw_files (bool, optional): whether remove raw images.
            Defaults to False.
        padding (int, optional): width of pixels between videos.
            Defaults to 0.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        None
    """
    lowercase = string.ascii_lowercase
    assert len(array) == 2
    assert (array[0] * array[1]) >= len(input_path_list)
    for path in input_path_list:
        check_input_path(
            path,
            allowed_suffix=['.gif', '.mp4'],
            tag='input video',
            path_type='file')
    prepare_output_path(
        output_path,
        allowed_suffix=['.gif', '.mp4'],
        tag='output video',
        path_type='file',
        overwrite=True)

    command = ['ffmpeg']
    height, width = resolution
    scale_command = []
    for index, vid_file in enumerate(input_path_list):
        command.append('-i')
        command.append(vid_file)
        scale_command.append(
            '[%d:v]scale=%d:%d:force_original_aspect_ratio=0[v%d];' %
            (index, width, height, index))

    scale_command = ' '.join(scale_command)
    pad_command = '[v%d]pad=%d:%d[%s];' % (0, width * array[1] + padding *
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

        pad_command += '[%s][v%d]overlay=%d:%d' % (lowercase[index - 1], index,
                                                   pad_width, pad_height)
        if index != len(input_path_list) - 1:
            pad_command += '[%s];' % lowercase[index]

    command += [
        '-filter_complex',
        '%s%s' % (scale_command, pad_command), '-loglevel', 'error', '-y',
        output_path
    ]
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    if remove_raw_files:
        command = ['rm', '-f'] + input_path_list
        subprocess.call(command)


def temporal_concat_video(input_path_list: List[str],
                          output_path: str,
                          resolution: Union[Tuple[int, int],
                                            Tuple[float, float]] = (512, 512),
                          remove_raw_files: bool = False,
                          disable_log: bool = False) -> None:
    """Concat no matter videos or gifs into a temporal sequence, and save as a
    new video or gif file.

    Args:
        input_path_list (List[str]): list of input video paths.
        output_path (str): output video file path.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]]
            , optional): (height, width) of output].
            Defaults to (512,512).
        remove_raw_files (bool, optional): whether remove the input videos.
            Defaults to False.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        None.
    """
    for path in input_path_list:
        check_input_path(
            path,
            allowed_suffix=['.gif', '.mp4'],
            tag='input video',
            path_type='file')
    prepare_output_path(
        output_path,
        allowed_suffix=['.gif', '.mp4'],
        tag='output video',
        path_type='file',
        overwrite=True)

    height, width = resolution
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
        '%s%sconcat=n=%d:v=1:a=0[v]' %
        (scale_command, concat_command, len(input_path_list)), '-loglevel',
        'error', '-map', '[v]', '-c:v', 'libx264', '-y', output_path
    ]
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    if remove_raw_files:
        command = ['rm'] + input_path_list
        subprocess.call(command)


def compress_video(input_path: str,
                   output_path: str,
                   compress_rate: int = 1,
                   down_sample_scale: Union[float, int] = 1,
                   fps: int = 30,
                   disable_log: bool = False) -> None:
    """Compress a video file.

    Args:
        input_path (str): input video file path.
        output_path (str): output video file path.
        compress_rate (int, optional): compress rate, influents the bit rate.
            Defaults to 1.
        down_sample_scale (Union[float, int], optional): spatial down sample
            scale. Defaults to 1.
        fps (int, optional): Frames per second. Defaults to 30.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
    Raises:
        FileNotFoundError: check the input path.
        FileNotFoundError: check the output path.

    Returns:
        None.
    """
    input_pathinfo = Path(input_path)

    check_input_path(
        input_path,
        allowed_suffix=['.gif', '.mp4'],
        tag='input video',
        path_type='file')
    prepare_output_path(
        output_path,
        allowed_suffix=['.gif', '.mp4'],
        tag='output video',
        path_type='file',
        overwrite=True)

    info = vid_info_reader(input_path)

    width = int(info['width'])
    height = int(info['height'])
    bit_rate = int(info['bit_rate'])
    duration = float(info['duration'])
    if (output_path == input_path) or (not output_path):
        temp_outpath = os.path.join(
            os.path.abspath(input_pathinfo.parent),
            'temp_file' + input_pathinfo.suffix)
    else:
        temp_outpath = output_path
    new_width = int(width / down_sample_scale)
    new_width += new_width % 2
    new_height = int(height / down_sample_scale)
    new_height += new_height % 2
    command = [
        'ffmpeg', '-y', '-r',
        str(info['r_frame_rate']), '-i', input_path, '-loglevel', 'error',
        '-b:v', f'{bit_rate / (compress_rate * down_sample_scale)}', '-r',
        f'{fps}', '-t', f'{duration}', '-s',
        '%dx%d' % (new_width, new_height), temp_outpath
    ]
    if not disable_log:
        print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    if (output_path == input_path) or (not output_path):
        subprocess.call(['mv', '-f', temp_outpath, input_path])


def pad_for_libx264(image_array):
    """Pad zeros if width or height of image_array is not divisible by 2.
    Otherwise you will get.

    \"[libx264 @ 0x1b1d560] width not divisible by 2 \"

    Args:
        image_array (np.ndarray):
            Image or images load by cv2.imread().
            Possible shapes:
            1. [height, width]
            2. [height, width, channels]
            3. [images, height, width]
            4. [images, height, width, channels]

    Returns:
        np.ndarray:
            A image with both edges divisible by 2.
    """
    if image_array.ndim == 2 or \
            (image_array.ndim == 3 and image_array.shape[2] == 3):
        hei_index = 0
        wid_index = 1
    elif image_array.ndim == 4 or \
            (image_array.ndim == 3 and image_array.shape[2] != 3):
        hei_index = 1
        wid_index = 2
    else:
        return image_array
    hei_pad = image_array.shape[hei_index] % 2
    wid_pad = image_array.shape[wid_index] % 2
    if hei_pad + wid_pad > 0:
        pad_width = []
        for dim_index in range(image_array.ndim):
            if dim_index == hei_index:
                pad_width.append((0, hei_pad))
            elif dim_index == wid_index:
                pad_width.append((0, wid_pad))
            else:
                pad_width.append((0, 0))
        values = 0
        image_array = \
            np.pad(image_array,
                   pad_width,
                   mode='constant', constant_values=values)
    return image_array
