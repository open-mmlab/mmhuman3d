import logging
import pickle
from typing import Any, Union

import numpy as np
from mmcv.utils import print_log

from mmhuman3d.data.data_structures.human_data import HumanData, _HumanData
from mmhuman3d.utils.path_utils import (
    Existence,
    check_path_existence,
    check_path_suffix,
)

_MultiHumanData_SUPPORTED_KEYS = HumanData.SUPPORTED_KEYS.copy()
_MultiHumanData_SUPPORTED_KEYS.update(
    {'frame_range': {
        'type': np.ndarray,
        'shape': (-1, 2),
        'dim': 0
    }})


class MultiHumanData(HumanData):
    SUPPORTED_KEYS = _MultiHumanData_SUPPORTED_KEYS

    def __new__(cls: _HumanData, *args: Any, **kwargs: Any) -> _HumanData:
        """New an instance of MultiHumanData.

        Args:
            cls (MultiHumanData): MultiHumanData class.

        Returns:
            HumanData: An instance of Hu
        """
        ret_human_data = super().__new__(cls, args, kwargs)
        setattr(ret_human_data, '__instance_num__', -1)
        return ret_human_data

    def load(self, npz_path: str):
        """Load data from npz_path and update them to self.

        Args:
            npz_path (str):
                Path to a dumped npz file.
        """
        supported_keys = self.__class__.SUPPORTED_KEYS
        with np.load(npz_path, allow_pickle=True) as npz_file:
            tmp_data_dict = dict(npz_file)
            for key, value in list(tmp_data_dict.items()):
                if isinstance(value, np.ndarray) and\
                        len(value.shape) == 0:
                    # value is not an ndarray before dump
                    value = value.item()
                elif key in supported_keys and\
                        type(value) != supported_keys[key]['type']:
                    value = supported_keys[key]['type'](value)
                if value is None:
                    tmp_data_dict.pop(key)
                elif key == '__key_strict__' or \
                        key == '__data_len__' or\
                        key == '__instance_num__' or\
                        key == '__keypoints_compressed__':
                    self.__setattr__(key, value)
                    # pop the attributes to keep dict clean
                    tmp_data_dict.pop(key)
                elif key == 'bbox_xywh' and value.shape[1] == 4:
                    value = np.hstack([value, np.ones([value.shape[0], 1])])
                    tmp_data_dict[key] = value
                else:
                    tmp_data_dict[key] = value
            self.update(tmp_data_dict)
            self.__set_default_values__()

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
            '__key_strict__': self.__key_strict__,
            '__data_len__': self.__data_len__,
            '__instance_num__': self.__instance_num__,
            '__keypoints_compressed__': self.__keypoints_compressed__,
        }
        dict_to_dump.update(self)
        np.savez_compressed(npz_path, **dict_to_dump)

    def dump_by_pickle(self, pkl_path: str, overwrite: bool = True) -> None:
        """Dump keys and items to a pickle file. It's a secondary dump method,
        when a HumanData instance is too large to be dumped by self.dump()

        Args:
            pkl_path (str):
                Path to a dumped pickle file.
            overwrite (bool, optional):
                Whether to overwrite if there is already a file.
                Defaults to True.

        Raises:
            ValueError:
                npz_path does not end with '.pkl'.
            FileExistsError:
                When overwrite is False and file exists.
        """
        if not check_path_suffix(pkl_path, ['.pkl']):
            raise ValueError('Not an pkl file.')
        if not overwrite:
            if check_path_existence(pkl_path, 'file') == Existence.FileExist:
                raise FileExistsError
        dict_to_dump = {
            '__key_strict__': self.__key_strict__,
            '__data_len__': self.__data_len__,
            '__instance_num__': self.__instance_num__,
            '__keypoints_compressed__': self.__keypoints_compressed__,
        }
        dict_to_dump.update(self)
        with open(pkl_path, 'wb') as f_writeb:
            pickle.dump(
                dict_to_dump, f_writeb, protocol=pickle.HIGHEST_PROTOCOL)

    def load_by_pickle(self, pkl_path: str) -> None:
        """Load data from pkl_path and update them to self.

        When a HumanData Instance was dumped by
        self.dump_by_pickle(), use this to load.
        Args:
            npz_path (str):
                Path to a dumped npz file.
        """
        with open(pkl_path, 'rb') as f_readb:
            tmp_data_dict = pickle.load(f_readb)
            for key, value in list(tmp_data_dict.items()):
                if value is None:
                    tmp_data_dict.pop(key)
                elif key == '__key_strict__' or \
                        key == '__data_len__' or\
                        key == '__instance_num__' or\
                        key == '__keypoints_compressed__':
                    self.__setattr__(key, value)
                    # pop the attributes to keep dict clean
                    tmp_data_dict.pop(key)
                elif key == 'bbox_xywh' and value.shape[1] == 4:
                    value = np.hstack([value, np.ones([value.shape[0], 1])])
                    tmp_data_dict[key] = value
                else:
                    tmp_data_dict[key] = value
            self.update(tmp_data_dict)
            self.__set_default_values__()

    @property
    def instance_num(self) -> int:
        """Get how many human are there in this MultiHumanData instance. In
        MuliHumanData, an image may have multiple corresponding human
        instances.

        Returns:
            int:
                Number of human instance related to this instance.
        """
        return self.__instance_num__

    @instance_num.setter
    def instance_num(self, value: int):
        """Set the human instance num of this MultiHumanData instance.

        Args:
            value (int):
                Number of human instance related to this instance.
        """
        self.__instance_num__ = value

    def get_slice(self,
                  arg_0: int,
                  arg_1: Union[int, Any] = None,
                  step: int = 1) -> _HumanData:
        """Slice all sliceable values along major_dim dimension.

        Args:
            arg_0 (int):
                When arg_1 is None, arg_0 is stop and start=0.
                When arg_1 is not None, arg_0 is start.
            arg_1 (Union[int, Any], optional):
                None or where to stop.
                Defaults to None.
            step (int, optional):
                Length of step. Defaults to 1.

        Returns:
            MultiHumanData:
                A new MultiHumanData instance with sliced values.
        """
        ret_human_data = \
            MultiHumanData.new(key_strict=self.get_key_strict())
        if arg_1 is None:
            start = 0
            stop = arg_0
        else:
            start = arg_0
            stop = arg_1
        slice_index = slice(start, stop, step)
        dim_dict = self.__get_slice_dim__()

        for key, dim in dim_dict.items():
            if key == 'frame_range':
                # primary index
                frame_range = None
            else:
                # index according to frame_range
                frame_range = self.get_raw_value('frame_range')

            # keys not expected be sliced
            if dim is None:
                ret_human_data[key] = self[key]
            elif isinstance(dim, dict):
                value_dict = self.get_raw_value(key)
                sliced_dict = {}
                for sub_key in value_dict.keys():
                    sub_value = value_dict[sub_key]
                    if dim[sub_key] is None:
                        sliced_dict[sub_key] = sub_value
                    else:
                        sub_dim = dim[sub_key]
                        sliced_sub_value = \
                            MultiHumanData.__get_sliced_result__(
                                sub_value, sub_dim, slice_index, frame_range)
                        sliced_dict[sub_key] = sliced_sub_value
                ret_human_data[key] = sliced_dict
            else:
                value = self[key]
                sliced_value = \
                    MultiHumanData.__get_sliced_result__(
                        value, dim, slice_index, frame_range)
                ret_human_data[key] = sliced_value
        # check keypoints compressed
        if self.check_keypoints_compressed():
            ret_human_data.compress_keypoints_by_mask()
        return ret_human_data

    def __get_slice_dim__(self) -> dict:
        """For each key in this HumanData, get the dimension for slicing. 0 for
        default, if no other value specified.

        Returns:
            dict:
                Keys are self.keys().
                Values indicate where to slice.
                None for not expected to be sliced or
                failed.
        """
        supported_keys = self.__class__.SUPPORTED_KEYS
        ret_dict = {}
        for key in self.keys():
            # keys not expected be sliced
            if key in supported_keys and \
                    'dim' in supported_keys[key] and \
                    supported_keys[key]['dim'] is None:
                ret_dict[key] = None
            else:
                value = self[key]
                if isinstance(value, dict) and len(value) > 0:
                    ret_dict[key] = {}
                    for sub_key in value.keys():
                        try:
                            sub_value_len = len(value[sub_key])
                            if sub_value_len != self.instance_num and \
                                    sub_value_len != self.data_len:
                                ret_dict[key][sub_key] = None
                            elif 'dim' in value:
                                ret_dict[key][sub_key] = value['dim']
                            else:
                                ret_dict[key][sub_key] = 0
                        except TypeError:
                            ret_dict[key][sub_key] = None
                    continue
                # instance cannot be sliced without len method
                try:
                    value_len = len(value)
                except TypeError:
                    ret_dict[key] = None
                    continue
                # slice on dim 0 by default
                slice_dim = 0
                if key in supported_keys and \
                        'dim' in supported_keys[key]:
                    slice_dim = \
                        supported_keys[key]['dim']
                data_len = value_len if slice_dim == 0 \
                    else value.shape[slice_dim]
                # dim not for slice
                if data_len != self.__instance_num__ and \
                        data_len != self.__data_len__:
                    ret_dict[key] = None
                    continue
                else:
                    ret_dict[key] = slice_dim
        return ret_dict

    # TODO: to support cache

    def __check_value_len__(self, key: Any, val: Any) -> bool:
        """Check whether the temporal length of val matches other values.

        Args:
            key (Any):
                Key in MultiHumanData.
            val (Any):
                Value to the key.

        Returns:
            bool:
                If temporal dim is defined and temporal length doesn't match,
                return False.
                Else return True.
        """
        ret_bool = True
        supported_keys = self.__class__.SUPPORTED_KEYS

        # MultiHumanData
        instance_num = 0
        if key == 'frame_range':
            for frame_range in val:
                instance_num += (frame_range[-1] - frame_range[0])

            if self.instance_num == -1:
                # init instance_num for multi_human_data
                self.instance_num = instance_num
            elif self.instance_num != instance_num:
                ret_bool = False

            data_len = len(val)
            if self.data_len == -1:
                # init data_len
                self.data_len = data_len
            elif self.data_len == self.instance_num:
                # update data_len
                self.data_len = data_len
            elif self.data_len != self.instance_num:
                ret_bool = False

        # check definition
        elif key in supported_keys:
            # check data length
            if 'dim' in supported_keys[key] and \
                    supported_keys[key]['dim'] is not None:
                val_slice_dim = supported_keys[key]['dim']
                if supported_keys[key]['type'] == dict:
                    slice_key = supported_keys[key]['slice_key']
                    val_data_len = val[slice_key].shape[val_slice_dim]
                else:
                    val_data_len = val.shape[val_slice_dim]

                if self.instance_num < 0:
                    # Init instance_num for HumanData,
                    # which is equal to data_len.
                    self.instance_num = val_data_len
                else:
                    # check if val_data_len matches recorded instance_num
                    if self.instance_num != val_data_len:
                        ret_bool = False

                if self.data_len < 0:
                    # init data_len for HumanData, it's equal to
                    # instance_num.
                    # If it's MultiHumanData needs to be updated
                    self.data_len = val_data_len

        if not ret_bool:
            err_msg = 'Data length check Failed:\n'
            err_msg += f'key={str(key)}\n'
            if self.data_len != self.instance_num:
                err_msg += f'val\'s instance_num={val_data_len}\n'
                err_msg += f'expected instance_num={self.instance_num}\n'
            print_log(
                msg=err_msg, logger=self.__class__.logger, level=logging.ERROR)
        return ret_bool

    def __set_default_values__(self) -> None:
        """For older versions of HumanData, call this method to apply missing
        values (also attributes).

        Note:
        1. Older HumanData doesn't define `data_len`.
        2. In the newer HumanData, `data_len` equals the `instances_num`.
        3. In MultiHumanData, `instance_num` equals instances num,
            and `data_len` equals frames num.
        """
        supported_keys = self.__class__.SUPPORTED_KEYS
        if 'frame_range' not in self:
            # the loaded file is not multi_human_data
            for key in supported_keys:
                if key in self and \
                        'dim' in supported_keys[key] and\
                        supported_keys[key]['dim'] is not None:
                    if 'slice_key' in supported_keys[key] and\
                            supported_keys[key]['type'] == dict:
                        sub_key = supported_keys[key]['slice_key']
                        slice_dim = supported_keys[key]['dim']
                        self.instance_num = self[key][sub_key].shape[slice_dim]
                    else:
                        slice_dim = supported_keys[key]['dim']
                        self.instance_num = self[key].shape[slice_dim]

                    # convert HumanData to MultiHumanData
                    self.data_len = self.instance_num
                    frame_range =  \
                        [[i, i + 1] for i in range(self.data_len)]
                    self['frame_range'] = np.array(frame_range)
                    break

        for key in list(self.keys()):
            convention_key = f'{key}_convention'
            if key.startswith('keypoints') and \
                    not key.endswith('_mask') and \
                    not key.endswith('_convention') and \
                    convention_key not in self:
                self[convention_key] = 'human_data'

    @classmethod
    def __get_sliced_result__(
            cls,
            input_data: Union[np.ndarray, list, tuple],
            slice_dim: int,
            slice_range: slice,
            frame_index: list = None) -> Union[np.ndarray, list, tuple]:

        if frame_index is not None:
            slice_data = []
            # primary index
            for frame_range in frame_index[slice_range]:
                slice_index = slice(frame_range[0], frame_range[-1], 1)
                slice_result = \
                    HumanData.__get_sliced_result__(
                        input_data,
                        slice_dim,
                        slice_index)
                for element in slice_result:
                    slice_data.append(element)
            if isinstance(input_data, np.ndarray):
                slice_data = np.array(slice_data)
            else:
                slice_data = type(input_data)(slice_data)
        else:
            slice_data = \
                HumanData.__get_sliced_result__(
                    input_data,
                    slice_dim,
                    slice_range)
        return slice_data
