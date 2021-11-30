import copy
from typing import Optional, Union

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from skimage.util.shape import view_as_windows

from .builder import DATASETS
from .human_image_dataset import HumanImageDataset


def get_vid_name(image_path: str):
    """Get base_dir of the given path."""
    content = image_path.split('/')
    vid_name = '/'.join(content[:-1])
    return vid_name


def split_into_chunks(data_infos: list, seq_len: int, stride: int,
                      test_mode: bool, only_vid_name: bool):
    """Split annotations into chunks.
    Adapted from https://github.com/mkocabas/VIBE
    Args:
        data_infos (list): parsed annotations.
        seq_len (int): the length of each chunk.
        stride (int): the interval between chunks.
        test_mode (bool): if test_mode is true, then an additional chunk
            will be added to cover all frames. Otherwise, last few frames
            will be dropped.
        only_vid_name (bool): if only_vid_name is true, image_path only
            contains the video name. Otherwise, image_path contains both
            video_name and frame index.

    Return:
        list:
            shape: [N, 4]. Each chunk contains four parameters: start_frame,
            end_frame, valid_start_frame, valid_end_frame. The last two
            parameters are used to suppress redundant frames.
    """
    vid_names = []
    for image_path in data_infos:
        if only_vid_name:
            vid_name = image_path
        else:
            vid_name = get_vid_name(image_path)
        vid_names.append(vid_name)
    vid_names = np.array(vid_names)
    video_start_end_indices = []

    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])

    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seq_len:
            continue
        chunks = view_as_windows(indexes, (seq_len, ), step=stride)
        start_finish = chunks[:, (0, -1, 0, -1)].tolist()
        video_start_end_indices += start_finish
        if chunks[-1][-1] < indexes[-1] and test_mode:
            start_frame = indexes[-1] - seq_len + 1
            end_frame = indexes[-1]
            valid_start_frame = chunks[-1][-1] + 1
            valid_end_frame = indexes[-1]
            extra_start_finish = [[
                start_frame, end_frame, valid_start_frame, valid_end_frame
            ]]
            video_start_end_indices += extra_start_finish

    return video_start_end_indices


@DATASETS.register_module()
class HumanVideoDataset(HumanImageDataset):
    """Human Video Dataset.

    Args:
        data_prefix (str): the prefix of data path.
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmhuman3d.datasets.pipelines`.
        dataset_name (str | None): the name of dataset. It is used to
            identify the type of evaluation metric. Default: None.
        seq_len (int, optional): the length of input sequence. Default: 16.
        overlap (float, optional): the overlap between different sequences.
            Default: 0
        only_vid_name (bool, optional): the format of image_path.
            If only_vid_name is true, image_path only contains the video
            name. Otherwise, image_path contains both video_name and frame
            index.
        body_model (dict | None, optional): the config for body model,
            which will be used to generate meshes and keypoints.
            Default: None.
        ann_file (str | None, optional): the annotation file. When ann_file
            is str, the subclass is expected to read from the ann_file. When
            ann_file is None, the subclass is expected to read according
            to data_prefix.
        convention (str, optional): keypoints convention. Keypoints will be
            converted from "human_data" to the given one.
            Default: "human_data"
        test_mode (bool, optional): in train mode or test mode. Default: False.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 dataset_name: str,
                 seq_len: Optional[int] = 16,
                 overlap: Optional[float] = 0.,
                 only_vid_name: Optional[bool] = False,
                 body_model: Optional[Union[dict, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 convention: Optional[str] = 'human_data',
                 test_mode: Optional[bool] = False):
        super(HumanVideoDataset, self).__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            dataset_name=dataset_name,
            body_model=body_model,
            convention=convention,
            ann_file=ann_file,
            test_mode=test_mode)
        self.seq_len = seq_len
        self.stride = int(seq_len * (1 - overlap))
        self.vid_indices = split_into_chunks(self.human_data['image_path'],
                                             self.seq_len, self.stride,
                                             test_mode, only_vid_name)
        self.vid_indices = np.array(self.vid_indices)

    def __len__(self):
        return len(self.vid_indices)

    def prepare_data(self, idx: int):
        """Prepare data for each chunk.

        Step 1: get annotation from each frame. Step 2: add metas of each
        chunk.
        """
        start_idx, end_idx = self.vid_indices[idx][:2]
        batch_results = []
        image_path = []
        for frame_idx in range(start_idx, end_idx + 1):
            frame_results = copy.deepcopy(self.prepare_raw_data(frame_idx))
            image_path.append(frame_results.pop('image_path'))
            if 'features' in self.human_data:
                frame_results['features'] = \
                     copy.deepcopy(self.human_data['features'][frame_idx])
            frame_results = self.pipeline(frame_results)
            batch_results.append(frame_results)
        video_results = {}
        for key in batch_results[0].keys():
            batch_anno = []
            for item in batch_results:
                batch_anno.append(item[key])
            if isinstance(batch_anno[0], torch.Tensor):
                batch_anno = torch.stack(batch_anno, dim=0)
            video_results[key] = batch_anno
        img_metas = {
            'frame_idx': self.vid_indices[idx],
            'image_path': image_path
        }
        video_results['img_metas'] = DC(img_metas, cpu_only=True)
        return video_results
