from typing import List

import numpy as np

from mmhuman3d.utils.path_utils import (
    Existence,
    check_path_existence,
    check_path_suffix,
)
from .human_data import HumanData


class HumanDataCacheReader():

    def __init__(self, npz_path: str):
        self.npz_path = npz_path
        npz_file = np.load(npz_path, allow_pickle=True)
        self.slice_size = npz_file['slice_size'].item()
        self.data_len = npz_file['data_len'].item()
        self.keypoints_info = npz_file['keypoints_info'].item()
        self.non_sliced_data = None
        self.npz_file = None

    def __del__(self):
        if self.npz_file is not None:
            self.npz_file.close()

    def get_item(self, index, required_keys: List[str] = []):
        if self.npz_file is None:
            self.npz_file = np.load(self.npz_path, allow_pickle=True)
        cache_key = str(int(index / self.slice_size))
        base_data = self.npz_file[cache_key].item()
        base_data.update(self.keypoints_info)
        for key in required_keys:
            non_sliced_value = self.get_non_sliced_data(key)
            if isinstance(non_sliced_value, dict) and\
                    key in base_data and\
                    isinstance(base_data[key], dict):
                base_data[key].update(non_sliced_value)
            else:
                base_data[key] = non_sliced_value
        ret_human_data = HumanData.new(source_dict=base_data)
        # data in cache is compressed
        ret_human_data.__keypoints_compressed__ = True
        # set missing values and attributes by default method
        ret_human_data.__set_default_values__()
        return ret_human_data

    def get_non_sliced_data(self, key: str):
        if self.non_sliced_data is None:
            if self.npz_file is None:
                npz_file = np.load(self.npz_path, allow_pickle=True)
                self.non_sliced_data = npz_file['non_sliced_data'].item()
            else:
                self.non_sliced_data = self.npz_file['non_sliced_data'].item()
        return self.non_sliced_data[key]


class HumanDataCacheWriter():

    def __init__(self,
                 slice_size: int,
                 data_len: int,
                 keypoints_info: dict,
                 non_sliced_data: dict,
                 key_strict: bool = True):
        self.slice_size = slice_size
        self.data_len = data_len
        self.keypoints_info = keypoints_info
        self.non_sliced_data = non_sliced_data
        self.sliced_data = {}
        self.key_strict = key_strict

    def update_sliced_dict(self, sliced_dict):
        self.sliced_data.update(sliced_dict)

    def dump(self, npz_path: str, overwrite: bool = True):
        """Dump keys and items to an npz file.

        Args:
            npz_path (str):
                Path to a dumped npz file.
            overwrite (bool, optional):
                Whether to overwrite if there is already a file.
                Defaults to True.

        Raises:
            ValueError:
                npz_path does not end with '.npz'.
            FileExistsError:
                When overwrite is False and file exists.
        """
        if not check_path_suffix(npz_path, ['.npz']):
            raise ValueError('Not an npz file.')
        if not overwrite:
            if check_path_existence(npz_path, 'file') == Existence.FileExist:
                raise FileExistsError
        dict_to_dump = {
            'slice_size': self.slice_size,
            'data_len': self.data_len,
            'keypoints_info': self.keypoints_info,
            'non_sliced_data': self.non_sliced_data,
            'key_strict': self.key_strict,
        }
        dict_to_dump.update(self.sliced_data)
        np.savez_compressed(npz_path, **dict_to_dump)
