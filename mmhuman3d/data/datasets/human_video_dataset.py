import copy

import numpy as np
from skimage.util.shape import view_as_windows

from .builder import DATASETS
from .human_image_dataset import HumanImageDataset


def get_vid_name(image_path):
    content = image_path.split('/')
    vid_name = '/'.join(content[:-1])
    return vid_name


def split_into_chunks(data_infos, seqlen, stride, test_mode, only_vid_name):
    vid_names = []
    for item in data_infos:
        image_path = item['image_path']
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
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen, ), step=stride)
        start_finish = chunks[:, (0, -1, 0, -1)].tolist()
        video_start_end_indices += start_finish
        if chunks[-1][-1] < indexes[-1] and test_mode:
            start_frame = indexes[-1] - seqlen + 1
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

    def __init__(self,
                 data_prefix,
                 pipeline,
                 dataset_name,
                 seq_len,
                 overlap=0.,
                 only_vid_name=False,
                 smpl=None,
                 ann_file=None,
                 test_mode=False):
        super(HumanVideoDataset,
              self).__init__(data_prefix, pipeline, dataset_name, smpl,
                             ann_file, test_mode)
        self.seq_len = seq_len
        self.stride = int(seq_len * (1 - overlap))
        self.vid_indices = split_into_chunks(self.data_infos, self.seq_len,
                                             self.stride, test_mode,
                                             only_vid_name)
        self.vid_indices = np.array(self.vid_indices)
        data = np.load(self.ann_file, allow_pickle=True)
        try:
            self.features = data['features']
        except KeyError:
            self.features = None

    def __len__(self):
        return len(self.vid_indices)

    def prepare_data(self, idx):
        start_idx, end_idx = self.vid_indices[idx][:2]
        batch_results = []
        for frame_idx in range(start_idx, end_idx + 1):
            frame_results = copy.deepcopy(self.data_infos[frame_idx])
            if self.features is not None:
                frame_results['features'] = \
                     copy.deepcopy(self.features[frame_idx])
            batch_results.append(frame_results)
        video_results = {}
        for key in batch_results[0].keys():
            batch_anno = []
            for item in batch_results:
                batch_anno.append(item[key])
            if isinstance(batch_anno[0], np.ndarray):
                batch_anno = np.stack(batch_anno, axis=0)
            video_results[key] = batch_anno
        video_results['frame_idx'] = self.vid_indices[idx]
        return self.pipeline(video_results)
