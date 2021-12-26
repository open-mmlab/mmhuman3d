import logging
import pickle
from enum import Enum
from typing import Any, Optional, Type, TypeVar, Union, overload

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
_CPU_DEVICE = torch.device('cpu')

_HumanData_SUPPORTED_KEYS = {
    'image_path': {
        'type': list,
    },
    'bbox_xywh': {
        'type': np.ndarray,
        'shape': (-1, 5),
        'temporal_dim': 0
    },
    'config': {
        'type': str,
    },
    'keypoints2d': {
        'type': np.ndarray,
        'shape': (-1, -1, 3),
        'temporal_dim': 0
    },
    'keypoints3d': {
        'type': np.ndarray,
        'shape': (-1, -1, 4),
        'temporal_dim': 0
    },
    'smpl': {
        'type': dict,
    },
    'smplx': {
        'type': dict,
    },
    'meta': {
        'type': dict,
    },
    'keypoints2d_mask': {
        'type': np.ndarray,
        'shape': (-1, ),
        'temporal_dim': -1
    },
    'keypoints3d_mask': {
        'type': np.ndarray,
        'shape': (-1, ),
        'temporal_dim': -1
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

    def __new__(cls: Type[_T1], *args: Any, **kwargs: Any) -> _T1:
        """New an instance of HumanData.

        Args:
            cls (Type[_T1]): HumanData class.

        Returns:
            _T1: An instance of HumanData.
        """
        ret_human_data = super().__new__(cls, args, kwargs)
        setattr(ret_human_data, '__temporal_len__', -1)
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
    def fromfile(cls, npz_path: str):
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
    def new(cls, source_dict: dict = None, key_strict: bool = False):
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
                        key == '__temporal_len__' or\
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
            '__temporal_len__': self.__temporal_len__,
            '__keypoints_compressed__': self.__keypoints_compressed__,
        }
        dict_to_dump.update(self)
        np.savez_compressed(npz_path, **dict_to_dump)

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
    def get_temporal_slice(self, stop: int):
        """Slice [0, stop, 1] of all temporal values."""
        ...

    @overload
    def get_temporal_slice(self, start: int, stop: int):
        """Slice [start, stop, 1] of all temporal values."""
        ...

    @overload
    def get_temporal_slice(self, start: int, stop: int, step: int):
        """Slice [start, stop, step] of all temporal values."""
        ...

    def get_temporal_slice(self,
                           arg_0: int,
                           arg_1: Union[int, Any] = None,
                           step: int = 1):
        """Slice all temporal values along timeline dimension.

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
        ret_human_data = HumanData.new(key_strict=self.get_key_strict())
        if arg_1 is None:
            start = 0
            stop = arg_0
        else:
            start = arg_0
            stop = arg_1
        slice_index = slice(start, stop, step)
        supported_keys = self.__class__.SUPPORTED_KEYS
        for key in self.keys():
            if key in supported_keys and \
                    'temporal_dim' in supported_keys[key] and \
                    supported_keys[key]['temporal_dim'] >= 0:
                slice_list = []
                for dim_index in range(len(supported_keys[key]['shape'])):
                    if dim_index == supported_keys[key]['temporal_dim']:
                        slice_list.append(slice_index)
                    else:
                        slice_list.append(slice(None))
                raw_value = self.get_raw_value(key)
                sliced_value = raw_value[tuple(slice_list)]
                ret_human_data.set_raw_value(key, sliced_value)
            elif key == 'image_path' and \
                    len(self[key]) == self.__temporal_len__:
                ret_human_data.set_raw_value(
                    key,
                    self.get_raw_value(key)[slice_index])
            else:
                ret_human_data.set_raw_value(key, self.get_raw_value(key))
        return ret_human_data

    def __setitem__(self, key: _KT, val: _VT):
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

    def pop_unsupported_items(self):
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
            self.__check_value_temporal__(key, val)
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
            err_msg = 'Type check Failed:\n'
            err_msg += f'key={str(key)}\n'
            err_msg += f'type(val)={type(val)}\n'
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
            err_msg = 'Shape check Failed:\n'
            err_msg += f'key={str(key)}\n'
            err_msg += f'val.shape={val_shape}\n'
            print_log(
                msg=err_msg, logger=self.__class__.logger, level=logging.ERROR)
        return ret_bool

    @property
    def temporal_len(self) -> int:
        """Get the temporal length of this HumanData instance.

        Returns:
            int:
                Number of frames related to this instance.
        """
        return self.__temporal_len__

    @temporal_len.setter
    def temporal_len(self, value: int):
        """Set the temporal length of this HumanData instance.

        Args:
            value (int):
                Number of frames related to this instance.
        """
        self.__temporal_len__ = value

    def __check_value_temporal__(self, key: Any, val: Any) -> bool:
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
            if 'temporal_dim' in supported_keys[key] and \
                    supported_keys[key]['temporal_dim'] >= 0:
                val_temporal_dim = supported_keys[key]['temporal_dim']
                val_temporal_len = val.shape[val_temporal_dim]
                if self.temporal_len < 0:
                    # no temporal_len yet, assign a new one
                    self.temporal_len = val_temporal_len
                else:
                    # check if val_temporal_len matches recorded temporal_len
                    if self.temporal_len != val_temporal_len:
                        ret_bool = False
        if not ret_bool:
            err_msg = 'Temporal check Failed:\n'
            err_msg += f'key={str(key)}\n'
            err_msg += f'val\'s temporal_len={val_temporal_len}\n'
            print_log(
                msg=err_msg, logger=self.__class__.logger, level=logging.ERROR)
        return ret_bool

    def compress_keypoints_by_mask(self):
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

    def decompress_keypoints(self):
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

    def dump_by_pickle(self, pkl_path: str, overwrite: bool = True):
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
            '__temporal_len__': self.__temporal_len__,
            '__keypoints_compressed__': self.__keypoints_compressed__,
        }
        dict_to_dump.update(self)
        with open(pkl_path, 'wb') as f_writeb:
            pickle.dump(
                dict_to_dump, f_writeb, protocol=pickle.HIGHEST_PROTOCOL)

    def load_by_pickle(self, pkl_path: str):
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
                        key == '__temporal_len__' or\
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
        temporal_len, _, dim = compressed_array.shape
        mask_len = mask_array.shape[0]
        ret_value = np.zeros(
            shape=[temporal_len, mask_len, dim], dtype=compressed_array.dtype)
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
