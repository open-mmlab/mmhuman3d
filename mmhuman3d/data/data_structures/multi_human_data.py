import logging
from enum import Enum
from typing import Any, TypeVar, Union

import numpy as np
from mmcv.utils import print_log

from mmhuman3d.data.data_structures.human_data import HumanData

# In T = TypeVar('T'), T can be anything.
# See definition of typing.TypeVar for details.
_HumanData = TypeVar('_HumanData')

_MultiHumanData_SUPPORTED_KEYS = HumanData().SUPPORTED_KEYS.update(
    {'optional': {
        'type': dict,
        'dim': None
    }})


class _KeyCheck(Enum):
    PASS = 0
    WARN = 1
    ERROR = 2


class MultiHumanData(HumanData):
    SUPPORTED_KEYS = _MultiHumanData_SUPPORTED_KEYS

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
        frame_range = self.get_raw_value('optional')['frame_range']
        for key, dim in dim_dict.items():
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
        return ret_human_data

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
        all_data_len = 0
        if 'optional' in self and \
                'frame_range' in self['optional']:
            for frame_range in self['optional']['frame_range']:
                # no data_len yet, assign a new one
                if self.data_len < 0:
                    self.data_len = len(self['optional']['frame_range'])
                all_data_len += len(frame_range)

        # check definition
        if key in supported_keys:
            # check temporal length
            if 'dim' in supported_keys[key] and \
                    supported_keys[key]['dim'] is not None:
                val_slice_dim = supported_keys[key]['dim']
                if supported_keys[key]['type'] == dict:
                    slice_key = supported_keys[key]['slice_key']
                    val_data_len = val[slice_key].shape[val_slice_dim]
                else:
                    val_data_len = val.shape[val_slice_dim]
                # all_data_len indicate the human_data type.
                # if all_data_len>0, it's the multi_human_data.
                if all_data_len > 0 and \
                        all_data_len != val_data_len:
                    ret_bool = False
                elif all_data_len == 0 and \
                        self.data_len < 0:
                    # no data_len yet, assign a new one
                    self.data_len = val_data_len
                elif all_data_len == 0 and \
                        self.data_len != val_data_len:
                    all_data_len = val_data_len
                    ret_bool = False

        if not ret_bool:
            err_msg = 'Temporal check Failed:\n'
            err_msg += f'key={str(key)}\n'
            err_msg += f'val\'s all_data_len={all_data_len}\n'
            err_msg += f'expected all_data_len={all_data_len}\n'
            print_log(
                msg=err_msg, logger=self.__class__.logger, level=logging.ERROR)
        return ret_bool

    def __set_default_values__(self) -> None:
        """For older versions of HumanData, call this method to apply missing
        values (also attributes).

        Note:
        1. Older HumanData doesn't define `data_len``.
        2. The value of `data_len`` defined in newer HumanData is equal to
           the `npz_len`.
        """
        supported_keys = self.__class__.SUPPORTED_KEYS
        for key in supported_keys:
            if key in self and \
                    'dim' in supported_keys[key] and\
                    supported_keys[key]['dim'] is not None:
                if 'slice_key' in supported_keys[key] and\
                        supported_keys[key]['type'] == dict:
                    sub_key = supported_keys[key]['slice_key']
                    slice_dim = supported_keys[key]['dim']
                    npz_len = self[key][sub_key].shape[slice_dim]
                else:
                    slice_dim = supported_keys[key]['dim']
                    npz_len = self[key].shape[slice_dim]
                break

        if self.__data_len__ == -1 or \
                self.__data_len__ == npz_len:
            multi_human_flag = False
        else:
            multi_human_flag = True

        # if the loaded npz file is HumanData,
        # create the `frame_range` according to
        # `npz_len`(data_len in HumanData)
        if not multi_human_flag:
            self['optional']['frame_range'] = \
                [[i, i + 1] for i in range(self.npz_len)]
            # for older version
            if self.__data_len__ == -1:
                self.__data_len__ = npz_len

        for key in list(self.keys()):
            convention_key = f'{key}_convention'
            if key.startswith('keypoints') and \
                    not key.endswith('_mask') and \
                    not key.endswith('_convention') and \
                    convention_key not in self:
                self[convention_key] = 'human_data'

    @classmethod
    def __get_sliced_result__(
            cls, input_data: Union[np.ndarray, list,
                                   tuple], slice_dim: int, slice_range: slice,
            frame_index: list) -> Union[np.ndarray, list, tuple]:
        slice_data = []
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
        return slice_data
