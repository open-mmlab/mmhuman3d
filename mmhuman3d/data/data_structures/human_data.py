import logging
import pickle
from enum import Enum
from math import ceil
from typing import Any, List, Optional, TypeVar, Union, overload

import numpy as np
import torch
from mmcv.utils import print_log

from mmhuman3d.utils.path_utils import (
    Existence,
    check_path_existence,
    check_path_suffix,
)

# In T = TypeVar('T'), T can be anything.
# See definition of typing.TypeVar for details.
_T1 = TypeVar('_T1')
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')
_HumanData = TypeVar('_HumanData')
_CPU_DEVICE = torch.device('cpu')

_HumanData_SUPPORTED_KEYS = {
    'image_path': {
        'type': list,
    },
    'image_id': {
        'type': list,
    },
    'bbox_xywh': {
        'type': np.ndarray,
        'shape': (-1, 5),
        'dim': 0
    },
    'config': {
        'type': str,
        'dim': None
    },
    'keypoints2d': {
        'type': np.ndarray,
        'shape': (-1, -1, 3),
        'dim': 0
    },
    'keypoints3d': {
        'type': np.ndarray,
        'shape': (-1, -1, 4),
        'dim': 0
    },
    'smpl': {
        'type': dict,
        'slice_key': 'betas',
        'dim': 0
    },
    'smplh': {
        'type': dict,
        'slice_key': 'betas',
        'dim': 0
    },
    'smplx': {
        'type': dict,
        'slice_key': 'betas',
        'dim': 0
    },
    'meta': {
        'type': dict,
    },
    'keypoints2d_mask': {
        'type': np.ndarray,
        'shape': (-1, ),
        'dim': None
    },
    'keypoints2d_convention': {
        'type': str,
        'dim': None
    },
    'keypoints3d_mask': {
        'type': np.ndarray,
        'shape': (-1, ),
        'dim': None
    },
    'keypoints3d_convention': {
        'type': str,
        'dim': None
    },
    'vertices': {
        'type': np.ndarray,
        'shape': (-1, ),
        'dim': None
    },
    'misc': {
        'type': dict,
    },
}


class _KeyCheck(Enum):
    PASS = 0
    WARN = 1
    ERROR = 2


class HumanData(dict):
    logger = None
    SUPPORTED_KEYS = _HumanData_SUPPORTED_KEYS
    WARNED_KEYS = []

    def __new__(cls: _HumanData, *args: Any, **kwargs: Any) -> _HumanData:
        """New an instance of HumanData.

        Args:
            cls (HumanData): HumanData class.

        Returns:
            HumanData: An instance of HumanData.
        """
        ret_human_data = super().__new__(cls, args, kwargs)
        setattr(ret_human_data, '__data_len__', -1)
        setattr(ret_human_data, '__key_strict__', False)
        setattr(ret_human_data, '__keypoints_compressed__', False)
        return ret_human_data

    @classmethod
    def set_logger(cls, logger: Union[logging.Logger, str, None] = None):
        """Set logger of HumanData class.

        Args:
            logger (logging.Logger | str | None, optional):
                The way to print summary.
                See `mmcv.utils.print_log()` for details.
                Defaults to None.
        """
        cls.logger = logger

    @classmethod
    def fromfile(cls, npz_path: str) -> _HumanData:
        """Construct a HumanData instance from an npz file.

        Args:
            npz_path (str):
                Path to a dumped npz file.

        Returns:
            HumanData:
                A HumanData instance load from file.
        """
        ret_human_data = cls()
        ret_human_data.load(npz_path)
        return ret_human_data

    @classmethod
    def new(cls,
            source_dict: dict = None,
            key_strict: bool = False) -> _HumanData:
        """Construct a HumanData instance from a dict.

        Args:
            source_dict (dict, optional):
                A dict with items in HumanData fashion.
                Defaults to None.
            key_strict (bool, optional):
                Whether to raise error when setting unsupported keys.
                Defaults to False.

        Returns:
            HumanData:
                A HumanData instance.
        """
        if source_dict is None:
            ret_human_data = cls()
        else:
            ret_human_data = cls(source_dict)
        ret_human_data.set_key_strict(key_strict)
        return ret_human_data

    def get_key_strict(self) -> bool:
        """Get value of attribute key_strict.

        Returns:
            bool:
                Whether to raise error when setting unsupported keys.
        """
        return self.__key_strict__

    def set_key_strict(self, value: bool):
        """Set value of attribute key_strict.

        Args:
            value (bool, optional):
                Whether to raise error when setting unsupported keys.
                Defaults to True.
        """
        former__key_strict__ = self.__key_strict__
        self.__key_strict__ = value
        if former__key_strict__ is False and \
                value is True:
            self.pop_unsupported_items()

    def check_keypoints_compressed(self) -> bool:
        """Check whether the keypoints are compressed.

        Returns:
            bool:
                Whether the keypoints are compressed.
        """
        return self.__keypoints_compressed__

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
            '__keypoints_compressed__': self.__keypoints_compressed__,
        }
        dict_to_dump.update(self)
        np.savez_compressed(npz_path, **dict_to_dump)

    def get_sliced_cache(self, slice_size=10) -> List:
        """Slice the whole HumanData into pieces for HumanDataCacheWriter.

        Args:
            slice_size (int, optional):
                The length of each unit in HumanData cache.
                Defaults to 10.

        Returns:
            List:
                Two dicts for HumanDataCacheWriter.
                Init HumanDataCacheWriter by HumanDataCacheWriter(**Returns[0])
                and set data by
                human_data_cache_writer.update_sliced_dict(Returns[1]).
        """
        keypoints_info = {}
        non_sliced_data = {}
        sliced_data = {}
        slice_num = ceil(self.__data_len__ / slice_size)
        for slice_index in range(slice_num):
            sliced_data[str(slice_index)] = {}
        dim_dict = self.__get_slice_dim__()
        for key, dim in dim_dict.items():
            # no dim to slice
            if dim is None:
                if key.startswith('keypoints') and\
                        (key.endswith('_mask') or
                         key.endswith('_convention')):
                    keypoints_info[key] = self[key]
                else:
                    non_sliced_data[key] = self[key]
            elif isinstance(dim, dict):
                value_dict = self.get_raw_value(key)
                non_sliced_sub_dict = {}
                for sub_key in value_dict.keys():
                    sub_value = value_dict[sub_key]
                    if dim[sub_key] is None:
                        non_sliced_sub_dict[sub_key] = sub_value
                    else:
                        sub_dim = dim[sub_key]
                        for slice_index in range(slice_num):
                            slice_start = slice_index * slice_size
                            slice_end = min((slice_index + 1) * slice_size,
                                            self.__data_len__)
                            slice_range = slice(slice_start, slice_end)
                            sliced_sub_value = \
                                HumanData.__get_sliced_result__(
                                    sub_value, sub_dim, slice_range
                                )
                            if key not in sliced_data[str(slice_index)]:
                                sliced_data[str(slice_index)][key] = {}
                            sliced_data[str(slice_index)][key][sub_key] = \
                                sliced_sub_value
                if len(non_sliced_sub_dict) > 0:
                    non_sliced_data[key] = non_sliced_sub_dict
            else:
                value = self.get_raw_value(key)
                # slice as ndarray
                if isinstance(value, np.ndarray):
                    slice_list = [
                        slice(None),
                    ] * len(value.shape)
                    for slice_index in range(slice_num):
                        slice_start = slice_index * slice_size
                        slice_end = min((slice_index + 1) * slice_size,
                                        self.__data_len__)
                        slice_list[dim] = slice(slice_start, slice_end)
                        sliced_value = value[tuple(slice_list)]
                        sliced_data[str(slice_index)][key] = sliced_value
                # slice as list/tuple
                else:
                    for slice_index in range(slice_num):
                        slice_start = slice_index * slice_size
                        slice_end = min((slice_index + 1) * slice_size,
                                        self.__data_len__)
                        sliced_value = value[slice(slice_start, slice_end)]
                        sliced_data[str(slice_index)][key] = sliced_value
        writer_args_dict = {
            'slice_size': slice_size,
            'keypoints_info': keypoints_info,
            'data_len': self.data_len,
            'non_sliced_data': non_sliced_data,
            'key_strict': self.get_key_strict()
        }
        return writer_args_dict, sliced_data

    def to(self,
           device: Optional[Union[torch.device, str]] = _CPU_DEVICE,
           dtype: Optional[torch.dtype] = None,
           non_blocking: Optional[bool] = False,
           copy: Optional[bool] = False,
           memory_format: Optional[torch.memory_format] = None) -> dict:
        """Convert values in numpy.ndarray type to torch.Tensor, and move
        Tensors to the target device. All keys will exist in the returned dict.

        Args:
            device (Union[torch.device, str], optional):
                A specified device. Defaults to CPU_DEVICE.
            dtype (torch.dtype, optional):
                The data type of the expected torch.Tensor.
                If dtype is None, it is decided according to numpy.ndarry.
                Defaults to None.
            non_blocking (bool, optional):
                When non_blocking, tries to convert asynchronously with
                respect to the host if possible, e.g.,
                converting a CPU Tensor with pinned memory to a CUDA Tensor.
                Defaults to False.
            copy (bool, optional):
                When copy is set, a new Tensor is created even when
                the Tensor already matches the desired conversion.
                No matter what value copy is, Tensor constructed from numpy
                will not share the same memory with the source numpy.ndarray.
                Defaults to False.
            memory_format (torch.memory_format, optional):
                The desired memory format of returned Tensor.
                Not supported by pytorch-cpu.
                Defaults to None.

        Returns:
            dict:
                A dict with all numpy.ndarray values converted into
                torch.Tensor and all Tensors moved to the target device.
        """
        ret_dict = {}
        for key in self.keys():
            raw_value = self.get_raw_value(key)
            tensor_value = None
            if isinstance(raw_value, np.ndarray):
                tensor_value = torch.from_numpy(raw_value).clone()
            elif isinstance(raw_value, torch.Tensor):
                tensor_value = raw_value
            if tensor_value is None:
                ret_dict[key] = raw_value
            else:
                if memory_format is None:
                    ret_dict[key] = \
                        tensor_value.to(device, dtype,
                                        non_blocking, copy)
                else:
                    ret_dict[key] = \
                        tensor_value.to(device, dtype,
                                        non_blocking, copy,
                                        memory_format=memory_format)
        return ret_dict

    def __getitem__(self, key: _KT) -> _VT:
        """Get value defined by HumanData. This function will be called by
        self[key]. In keypoints_compressed mode, if the key contains
        'keypoints', an array with zero-padding at absent keypoint will be
        returned. Call self.get_raw_value(k) to get value without padding.

        Args:
            key (_KT):
                Key in HumanData.

        Returns:
            _VT:
                Value to the key.
        """
        value = super().__getitem__(key)
        if self.__keypoints_compressed__:
            mask_key = f'{key}_mask'
            if key in self and \
                    isinstance(value, np.ndarray) and \
                    'keypoints' in key and \
                    mask_key in self:
                mask_array = np.asarray(super().__getitem__(mask_key))
                value = \
                    self.__class__.__add_zero_pad__(value, mask_array)
        return value

    def get_raw_value(self, key: _KT) -> _VT:
        """Get raw value from the dict. It acts the same as
        dict.__getitem__(k).

        Args:
            key (_KT):
                Key in dict.

        Returns:
            _VT:
                Value to the key.
        """
        value = super().__getitem__(key)
        return value

    def get_value_in_shape(self,
                           key: _KT,
                           shape: Union[list, tuple],
                           padding_constant: int = 0) -> np.ndarray:
        """Get value in a specific shape. For each dim, if the required shape
        is smaller than current shape, ndarray will be sliced. Otherwise, it
        will be padded with padding_constant at the end.

        Args:
            key (_KT):
                Key in dict. The value of this key must be
                an instance of numpy.ndarray.
            shape (Union[list, tuple]):
                Shape of the returned array. Its length
                must be equal to value.ndim. Set -1 for
                a dimension if you do not want to edit it.
            padding_constant (int, optional):
                The value to set the padded values for each axis.
                Defaults to 0.

        Raises:
            ValueError:
                A value in shape is neither positive integer nor -1.

        Returns:
            np.ndarray:
                An array in required shape.
        """
        value = self.get_raw_value(key)
        assert isinstance(value, np.ndarray)
        assert value.ndim == len(shape)
        pad_width_list = []
        slice_list = []
        for dim_index in range(len(shape)):
            if shape[dim_index] == -1:
                # no pad or slice
                pad_width_list.append((0, 0))
                slice_list.append(slice(None))
            elif shape[dim_index] > 0:
                # valid shape value
                wid = shape[dim_index] - value.shape[dim_index]
                if wid > 0:
                    pad_width_list.append((0, wid))
                else:
                    pad_width_list.append((0, 0))
                slice_list.append(slice(0, shape[dim_index]))
            else:
                # invalid
                raise ValueError
        pad_value = np.pad(
            value,
            pad_width=pad_width_list,
            mode='constant',
            constant_values=padding_constant)
        return pad_value[tuple(slice_list)]

    @overload
    def get_slice(self, stop: int):
        """Slice [0, stop, 1] of all sliceable values."""
        ...

    @overload
    def get_slice(self, start: int, stop: int):
        """Slice [start, stop, 1] of all sliceable values."""
        ...

    @overload
    def get_slice(self, start: int, stop: int, step: int):
        """Slice [start, stop, step] of all sliceable values."""
        ...

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
            HumanData:
                A new HumanData instance with sliced values.
        """
        ret_human_data = \
            HumanData.new(key_strict=self.get_key_strict())
        if arg_1 is None:
            start = 0
            stop = arg_0
        else:
            start = arg_0
            stop = arg_1
        slice_index = slice(start, stop, step)
        dim_dict = self.__get_slice_dim__()
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
                            HumanData.__get_sliced_result__(
                                sub_value, sub_dim, slice_index)
                        sliced_dict[sub_key] = sliced_sub_value
                ret_human_data[key] = sliced_dict
            else:
                value = self[key]
                sliced_value = \
                    HumanData.__get_sliced_result__(
                        value, dim, slice_index)
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
                            if 'dim' in value:
                                ret_dict[key][sub_key] = value['dim']
                            elif sub_value_len != self.__data_len__:
                                ret_dict[key][sub_key] = None
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
                if data_len != self.__data_len__:
                    ret_dict[key] = None
                    continue
                else:
                    ret_dict[key] = slice_dim
        return ret_dict

    def __setitem__(self, key: _KT, val: _VT) -> None:
        """Set self[key] to value. Only be called when using
        human_data[key] = val. Methods like update won't call __setitem__.
        In keypoints_compressed mode, if the key contains 'keypoints',
        and f'{key}_mask' is in self.keys(), invalid zeros
        will be removed before setting value.

        Args:
            key (_KT):
                Key in HumanData.
                Better be an element in HumanData.SUPPORTED_KEYS.
                If not, an Error will be raised in key_strict mode.
            val (_VT):
                Value to the key.

        Raises:
            KeyError:
                self.get_key_strict() is True and
                key cannot be found in
                HumanData.SUPPORTED_KEYS.
            ValueError:
                Value is supported but doesn't match definition.
            ValueError:
                self.check_keypoints_compressed() is True and
                mask of a keypoint item is missing.
        """
        self.__check_key__(key)
        self.__check_value__(key, val)
        # if it can be compressed by mask
        if self.__keypoints_compressed__:
            class_logger = self.__class__.logger
            if 'keypoints' in key and \
                    '_mask' in key:
                msg = 'Mask cannot be modified ' +\
                      'in keypoints_compressed mode.'
                print_log(msg=msg, logger=class_logger, level=logging.WARN)
                return
            elif isinstance(val, np.ndarray) and \
                    'keypoints' in key and \
                    '_mask' not in key:
                mask_key = f'{key}_mask'
                if mask_key in self:
                    mask_array = np.asarray(super().__getitem__(mask_key))
                    val = \
                        self.__class__.__remove_zero_pad__(val, mask_array)
                else:
                    msg = f'Mask for {key} has not been set.' +\
                        f' Please set {mask_key} before compression.'
                    print_log(
                        msg=msg, logger=class_logger, level=logging.ERROR)
                    raise ValueError
        dict.__setitem__(self, key, val)

    def set_raw_value(self, key: _KT, val: _VT) -> None:
        """Set the raw value of self[key] to val after key check. It acts the
        same as dict.__setitem__(self, key, val) if the key satisfied
        constraints.

        Args:
            key (_KT):
                Key in dict.
            val (_VT):
                Value to the key.

        Raises:
            KeyError:
                self.get_key_strict() is True and
                key cannot be found in
                HumanData.SUPPORTED_KEYS.
            ValueError:
                Value is supported but doesn't match definition.
        """
        self.__check_key__(key)
        self.__check_value__(key, val)
        dict.__setitem__(self, key, val)

    def pop_unsupported_items(self) -> None:
        """Find every item with a key not in HumanData.SUPPORTED_KEYS, and pop
        it to save memory."""
        for key in list(self.keys()):
            if key not in self.__class__.SUPPORTED_KEYS:
                self.pop(key)

    def __check_key__(self, key: Any) -> _KeyCheck:
        """Check whether the key matches definition in
        HumanData.SUPPORTED_KEYS.

        Args:
            key (Any):
                Key in HumanData.

        Returns:
            _KeyCheck:
                PASS, WARN or ERROR.

        Raises:
            KeyError:
                self.get_key_strict() is True and
                key cannot be found in
                HumanData.SUPPORTED_KEYS.
        """
        ret_key_check = _KeyCheck.PASS
        if self.get_key_strict():
            if key not in self.__class__.SUPPORTED_KEYS:
                ret_key_check = _KeyCheck.ERROR
        else:
            if key not in self.__class__.SUPPORTED_KEYS and \
                    key not in self.__class__.WARNED_KEYS:
                # log warning message at the first time
                ret_key_check = _KeyCheck.WARN
                self.__class__.WARNED_KEYS.append(key)
        if ret_key_check == _KeyCheck.ERROR:
            raise KeyError(self.__class__.__get_key_error_msg__(key))
        elif ret_key_check == _KeyCheck.WARN:
            class_logger = self.__class__.logger
            if class_logger == 'silent':
                pass
            else:
                print_log(
                    msg=self.__class__.__get_key_warn_msg__(key),
                    logger=class_logger,
                    level=logging.WARN)
        return ret_key_check

    def __check_value__(self, key: Any, val: Any) -> bool:
        """Check whether the value matches definition in
        HumanData.SUPPORTED_KEYS.

        Args:
            key (Any):
                Key in HumanData.
            val (Any):
                Value to the key.

        Returns:
            bool:
                True for matched, ortherwise False.

        Raises:
            ValueError:
                Value is supported but doesn't match definition.
        """
        ret_bool = self.__check_value_type__(key, val) and\
            self.__check_value_shape__(key, val) and\
            self.__check_value_len__(key, val)
        if not ret_bool:
            raise ValueError(self.__class__.__get_value_error_msg__())
        return ret_bool

    def __check_value_type__(self, key: Any, val: Any) -> bool:
        """Check whether the type of val matches definition in
        HumanData.SUPPORTED_KEYS.

        Args:
            key (Any):
                Key in HumanData.
            val (Any):
                Value to the key.

        Returns:
            bool:
                If type doesn't match, return False.
                Else return True.
        """
        ret_bool = True
        supported_keys = self.__class__.SUPPORTED_KEYS
        # check definition
        if key in supported_keys:
            # check type
            if type(val) != supported_keys[key]['type']:
                ret_bool = False
        if not ret_bool:
            expected_type = supported_keys[key]['type']
            err_msg = 'Type check Failed:\n'
            err_msg += f'key={str(key)}\n'
            err_msg += f'type(val)={type(val)}\n'
            err_msg += f'expected type={expected_type}\n'
            print_log(
                msg=err_msg, logger=self.__class__.logger, level=logging.ERROR)
        return ret_bool

    def __check_value_shape__(self, key: Any, val: Any) -> bool:
        """Check whether the shape of val matches definition in
        HumanData.SUPPORTED_KEYS.

        Args:
            key (Any):
                Key in HumanData.
            val (Any):
                Value to the key.

        Returns:
            bool:
                If expected shape is defined and doesn't match,
                return False.
                Else return True.
        """
        ret_bool = True
        supported_keys = self.__class__.SUPPORTED_KEYS
        # check definition
        if key in supported_keys:
            # check shape
            if 'shape' in supported_keys[key]:
                val_shape = val.shape
                for shape_ind in range(len(supported_keys[key]['shape'])):
                    # length not match
                    if shape_ind >= len(val_shape):
                        ret_bool = False
                        break
                    expect_val = supported_keys[key]['shape'][shape_ind]
                    # value not match
                    if expect_val > 0 and \
                            expect_val != val_shape[shape_ind]:
                        ret_bool = False
                        break
        if not ret_bool:
            expected_shape = str(supported_keys[key]['shape'])
            expected_shape = expected_shape.replace('-1', 'Any')
            err_msg = 'Shape check Failed:\n'
            err_msg += f'key={str(key)}\n'
            err_msg += f'val.shape={val_shape}\n'
            err_msg += f'expected shape={expected_shape}\n'
            print_log(
                msg=err_msg, logger=self.__class__.logger, level=logging.ERROR)
        return ret_bool

    @property
    def data_len(self) -> int:
        """Get the temporal length of this HumanData instance.

        Returns:
            int:
                Number of frames related to this instance.
        """
        return self.__data_len__

    @data_len.setter
    def data_len(self, value: int):
        """Set the temporal length of this HumanData instance.

        Args:
            value (int):
                Number of frames related to this instance.
        """
        self.__data_len__ = value

    def __check_value_len__(self, key: Any, val: Any) -> bool:
        """Check whether the temporal length of val matches other values.

        Args:
            key (Any):
                Key in HumanData.
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
                if self.data_len < 0:
                    # no data_len yet, assign a new one
                    self.data_len = val_data_len
                else:
                    # check if val_data_len matches recorded data_len
                    if self.data_len != val_data_len:
                        ret_bool = False
        if not ret_bool:
            err_msg = 'Temporal check Failed:\n'
            err_msg += f'key={str(key)}\n'
            err_msg += f'val\'s data_len={val_data_len}\n'
            err_msg += f'expected data_len={self.data_len}\n'
            print_log(
                msg=err_msg, logger=self.__class__.logger, level=logging.ERROR)
        return ret_bool

    def generate_mask_from_confidence(self, keys=None) -> None:
        """Generate mask from keypoints' confidence. Keypoints that have zero
        confidence in all occurrences will have a zero mask. Note that the last
        value of the keypoint is assumed to be confidence.

        Args:
            keys: None, str, or list of str.
                None: all keys with `keypoint` in it will have mask
                    generated from their confidence.
                str: key of the keypoint, the mask has name f'{key}_name'
                list of str: a list of keys of the keypoints.
                    Generate mask for multiple keypoints.
                Defaults to None.

        Returns:
            None

        Raises:
            KeyError:
                A key is not not found
        """
        if keys is None:
            keys = []
            for key in self.keys():
                val = self.get_raw_value(key)
                if isinstance(val, np.ndarray) and \
                        'keypoints' in key and \
                        '_mask' not in key:
                    keys.append(key)
        elif isinstance(keys, str):
            keys = [keys]
        elif isinstance(keys, list):
            for key in keys:
                assert isinstance(key, str)
        else:
            raise TypeError(f'`Keys` must be None, str, or list of str, '
                            f'got {type(keys)}.')

        update_dict = {}
        for kpt_key in keys:
            kpt_array = self.get_raw_value(kpt_key)
            num_joints = kpt_array.shape[-2]
            # if all conf of a joint are zero, this joint is masked
            joint_conf = kpt_array[..., -1].reshape(-1, num_joints)
            mask_array = (joint_conf > 0).astype(np.uint8).max(axis=0)
            assert len(mask_array) == num_joints
            # generate mask
            update_dict[f'{kpt_key}_mask'] = mask_array
        self.update(update_dict)

    def compress_keypoints_by_mask(self) -> None:
        """If a key contains 'keypoints', and f'{key}_mask' is in self.keys(),
        invalid zeros will be removed and f'{key}_mask' will be locked.

        Raises:
            KeyError:
                A key contains 'keypoints' has been found
                but its corresponding mask is missing.
        """
        assert self.__keypoints_compressed__ is False
        key_pairs = []
        for key in self.keys():
            mask_key = f'{key}_mask'
            val = self.get_raw_value(key)
            if isinstance(val, np.ndarray) and \
                    'keypoints' in key and \
                    '_mask' not in key:
                if mask_key in self:
                    key_pairs.append([key, mask_key])
                else:
                    msg = f'Mask for {key} has not been set.' +\
                        f'Please set {mask_key} before compression.'
                    raise KeyError(msg)
        compressed_dict = {}
        for kpt_key, mask_key in key_pairs:
            kpt_array = self.get_raw_value(kpt_key)
            mask_array = np.asarray(self.get_raw_value(mask_key))
            compressed_kpt = \
                self.__class__.__remove_zero_pad__(kpt_array, mask_array)
            compressed_dict[kpt_key] = compressed_kpt
        # set value after all pairs are compressed
        self.update(compressed_dict)
        self.__keypoints_compressed__ = True

    def decompress_keypoints(self) -> None:
        """If a key contains 'keypoints', and f'{key}_mask' is in self.keys(),
        invalid zeros will be inserted to the right places and f'{key}_mask'
        will be unlocked.

        Raises:
            KeyError:
                A key contains 'keypoints' has been found
                but its corresponding mask is missing.
        """
        assert self.__keypoints_compressed__ is True
        key_pairs = []
        for key in self.keys():
            mask_key = f'{key}_mask'
            val = self.get_raw_value(key)
            if isinstance(val, np.ndarray) and \
                    'keypoints' in key and \
                    '_mask' not in key:
                if mask_key in self:
                    key_pairs.append([key, mask_key])
                else:
                    class_logger = self.__class__.logger
                    msg = f'Mask for {key} has not been found.' +\
                        f'Please remove {key} before decompression.'
                    print_log(
                        msg=msg, logger=class_logger, level=logging.ERROR)
                    raise KeyError
        decompressed_dict = {}
        for kpt_key, mask_key in key_pairs:
            mask_array = np.asarray(self.get_raw_value(mask_key))
            compressed_kpt = self.get_raw_value(kpt_key)
            kpt_array = \
                self.__class__.__add_zero_pad__(compressed_kpt, mask_array)
            decompressed_dict[kpt_key] = kpt_array
        # set value after all pairs are decompressed
        self.update(decompressed_dict)
        self.__keypoints_compressed__ = False

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

    def __set_default_values__(self) -> None:
        """For older versions of HumanData, call this method to apply missing
        values (also attributes)."""
        supported_keys = self.__class__.SUPPORTED_KEYS
        if self.__data_len__ == -1:
            for key in supported_keys:
                if key in self and \
                        'dim' in supported_keys[key] and\
                        supported_keys[key]['dim'] is not None:
                    if 'slice_key' in supported_keys[key] and\
                            supported_keys[key]['type'] == dict:
                        sub_key = supported_keys[key]['slice_key']
                        slice_dim = supported_keys[key]['dim']
                        self.__data_len__ = \
                            self[key][sub_key].shape[slice_dim]
                    else:
                        slice_dim = supported_keys[key]['dim']
                        self.__data_len__ = self[key].shape[slice_dim]
                    break
        for key in list(self.keys()):
            convention_key = f'{key}_convention'
            if key.startswith('keypoints') and \
                    not key.endswith('_mask') and \
                    not key.endswith('_convention') and \
                    convention_key not in self:
                self[convention_key] = 'human_data'

    @classmethod
    def concatenate(cls, human_data_0: _HumanData,
                    human_data_1: _HumanData) -> _HumanData:
        """Concatenate two human_data. All keys will be kept it the returned
        human_data. If either value from human_data_0 or human_data_1 matches
        data_len from its HumanData, the two values will be concatenated as a
        single value. If not, postfix will be added to the key to specify
        source of the value.

        Args:
            human_data_0 (_HumanData)
            human_data_1 (_HumanData)

        Returns:
            _HumanData:
                A new human_data instance with all concatenated data.
        """
        ret_human_data = cls.new(key_strict=False)
        set_0 = set(human_data_0.keys())
        set_1 = set(human_data_1.keys())
        common_keys = set_0.intersection(set_1)
        dim_dict_0 = human_data_0.__get_slice_dim__()
        dim_dict_1 = human_data_1.__get_slice_dim__()
        for key in common_keys:
            value_0 = human_data_0[key]
            value_1 = human_data_1[key]
            # align type
            value_0 = list(value_0) if isinstance(value_0, tuple)\
                else value_0
            value_1 = list(value_1) if isinstance(value_1, tuple)\
                else value_1
            assert type(value_0) == type(value_1)
            # align convention
            if key.startswith('keypoints') and\
                    key.endswith('_convention'):
                assert value_0 == value_1
                ret_human_data[key] = value_0
                continue
            # mask_0 and mask_1
            elif key.startswith('keypoints') and\
                    key.endswith('_mask'):
                new_mask = value_0 * value_1
                ret_human_data[key] = new_mask
                continue
            # go through the sub dict
            if isinstance(value_0, dict):
                sub_dict = {}
                for sub_key, sub_value_0 in value_0.items():
                    # only found in value_0
                    if sub_key not in value_1:
                        sub_dict[sub_key] = sub_value_0
                    # found in both values
                    else:
                        sub_value_1 = value_1[sub_key]
                        concat_sub_dict = cls.__concat_value__(
                            key=sub_key,
                            value_0=sub_value_0,
                            dim_0=dim_dict_0[key][sub_key],
                            value_1=sub_value_1,
                            dim_1=dim_dict_1[key][sub_key])
                        sub_dict.update(concat_sub_dict)
                for sub_key, sub_value_1 in value_1.items():
                    if sub_key not in value_0:
                        sub_dict[sub_key] = sub_value_1
                ret_human_data[key] = sub_dict
            # try concat
            else:
                concat_dict = cls.__concat_value__(
                    key=key,
                    value_0=value_0,
                    dim_0=dim_dict_0[key],
                    value_1=value_1,
                    dim_1=dim_dict_1[key])
                ret_human_data.update(concat_dict)
        # check exclusive keys
        for key, value in human_data_0.items():
            if key not in common_keys:
                # value not for concat and slice
                if dim_dict_0[key] is None:
                    ret_human_data[key] = value
                # value aligned with data_len of HumanData_0
                else:
                    ret_human_data[f'{key}_0'] = value
        for key, value in human_data_1.items():
            if key not in common_keys:
                # same as above
                if dim_dict_1[key] is None:
                    ret_human_data[key] = value
                else:
                    ret_human_data[f'{key}_1'] = value
        return ret_human_data

    @classmethod
    def __concat_value__(cls, key: Any, value_0: Any, value_1: Any,
                         dim_0: Union[None, int], dim_1: Union[None,
                                                               int]) -> dict:
        """Concat two values from two different HumanData.

        Args:
            key (Any):
                The common key of the two values.
            value_0 (Any):
                Value from 0.
            value_1 (Any):
                Value from 1.
            dim_0 (Union[None, int]):
                The dim for concat and slice. None for N/A.
            dim_1 (Union[None, int]):
                The dim for concat and slice. None for N/A.

        Returns:
            dict:
                Dict for concatenated result.
        """
        ret_dict = {}
        if dim_0 is None or dim_1 is None:
            ret_dict[f'{key}_0'] = value_0
            ret_dict[f'{key}_1'] = value_1
        elif isinstance(value_0, list):
            ret_dict[key] = value_0 + value_1
        # elif isinstance(value_0, np.ndarray):
        else:
            ret_dict[key] = np.concatenate((value_0, value_1), axis=dim_0)
        return ret_dict

    @classmethod
    def __add_zero_pad__(cls, compressed_array: np.ndarray,
                         mask_array: np.ndarray) -> np.ndarray:
        """Pad zeros to a compressed keypoints array.

        Args:
            compressed_array (np.ndarray):
                A compressed keypoints array.
            mask_array (np.ndarray):
                The mask records compression relationship.

        Returns:
            np.ndarray:
                A keypoints array in full-size.
        """
        assert mask_array.sum() == compressed_array.shape[1]
        data_len, _, dim = compressed_array.shape
        mask_len = mask_array.shape[0]
        ret_value = np.zeros(
            shape=[data_len, mask_len, dim], dtype=compressed_array.dtype)
        valid_mask_index = np.where(mask_array == 1)[0]
        ret_value[:, valid_mask_index, :] = compressed_array
        return ret_value

    @classmethod
    def __remove_zero_pad__(cls, zero_pad_array: np.ndarray,
                            mask_array: np.ndarray) -> np.ndarray:
        """Remove zero-padding from a full-size keypoints array.

        Args:
            zero_pad_array (np.ndarray):
                A keypoints array in full-size.
            mask_array (np.ndarray):
                The mask records compression relationship.

        Returns:
            np.ndarray:
                A compressed keypoints array.
        """
        assert mask_array.shape[0] == zero_pad_array.shape[1]
        valid_mask_index = np.where(mask_array == 1)[0]
        ret_value = np.take(zero_pad_array, valid_mask_index, axis=1)
        return ret_value

    @classmethod
    def __get_key_warn_msg__(cls, key: Any) -> str:
        """Get the warning message when a key fails the check.

        Args:
            key (Any):
                The key with wrong.

        Returns:
            str:
                The warning message.
        """
        class_name = cls.__name__
        warn_message = \
            f'{key} is absent in' +\
            f' {class_name}.SUPPORTED_KEYS.\n'
        suggestion_message = \
            'Ignore this if you know exactly' +\
            ' what you are doing.\n' +\
            'Otherwise, Call self.set_key_strict(True)' +\
            ' to avoid wrong keys.\n'
        return warn_message + suggestion_message

    @classmethod
    def __get_key_error_msg__(cls, key: Any) -> str:
        """Get the error message when a key fails the check.

        Args:
            key (Any):
                The key with wrong.

        Returns:
            str:
                The error message.
        """
        class_name = cls.__name__
        absent_message = \
            f'{key} is absent in' +\
            f' {class_name}.SUPPORTED_KEYS.\n'
        suggestion_message = \
            'Call self.set_key_strict(False)' +\
            ' to allow unsupported keys.\n'
        return absent_message + suggestion_message

    @classmethod
    def __get_value_error_msg__(cls) -> str:
        """Get the error message when a value fails the check.

        Returns:
            str:
                The error message.
        """
        error_message = \
            'An supported value doesn\'t ' +\
            'match definition.\n'
        suggestion_message = \
            'See error log for details.\n'
        return error_message + suggestion_message

    @classmethod
    def __get_sliced_result__(
            cls, input_data: Union[np.ndarray, list, tuple], slice_dim: int,
            slice_range: slice) -> Union[np.ndarray, list, tuple]:
        """Slice input_data along slice_dim with slice_range.

        Args:
            input_data (Union[np.ndarray, list, tuple]):
                Data to be sliced.
            slice_dim (int):
                Dimension to be sliced.
            slice_range (slice):
                An instance of class slice.

        Returns:
            Union[np.ndarray, list, tuple]:
                A slice of input_data.
        """
        if isinstance(input_data, np.ndarray):
            slice_list = [
                slice(None),
            ] * len(input_data.shape)
            slice_list[slice_dim] = slice_range
            sliced_data = input_data[tuple(slice_list)]
        else:
            sliced_data = \
                input_data[slice_range]
        return sliced_data
