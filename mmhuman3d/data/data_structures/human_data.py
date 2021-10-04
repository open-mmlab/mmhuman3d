import logging
from enum import Enum
from typing import Any, Type, TypeVar, Union

import numpy as np
from mmcv.utils import print_log

from mmhuman3d.utils.path_utils import (
    Existence,
    check_path_existence,
    check_path_suffix,
)

_T1 = TypeVar('_T1')

_HumanData_SUPPORTED_KEYS = {
    'image_path': {
        'type': str,
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
                A dict with items in HumanData fasion.
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

    def load(self, npz_path: str):
        """Load data from npz_path and update them to self.

        Args:
            npz_path (str):
                Path to a dumped npz file.
        """
        with np.load(npz_path, allow_pickle=True) as npz_file:
            tmp_data_dict = dict(npz_file)
            for key, value in list(tmp_data_dict.items()):
                if isinstance(value, np.ndarray) and\
                        len(value.shape) == 0:
                    # value is not an ndarray before dump
                    value = value.item()
                if value is None:
                    tmp_data_dict.pop(key)
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
        np.savez_compressed(npz_path, **self)

    def __setitem__(self, key: Any, val: Any):
        """Set self[key] to value. Only be called when using
        human_data[key] = val. Methods like update won't call __setitem__.

        Args:
            key (Any):
                Key in HumanData.
                Better be an element in HumanData.SUPPORTED_KEYS.
                If not, an Error will be raised in key_strict mode.
            val (Any):
                Value to the key.

        Raises:
            KeyError:
                self.get_key_strict() is True and
                key cannot be found in
                HumanData.SUPPORTED_KEYS.
        """
        key_check = self.__check_key__(key)
        if key_check == _KeyCheck.ERROR:
            raise KeyError(self.__class__.__get_key_error_msg__(key))
        elif key_check == _KeyCheck.WARN:
            class_logger = self.__class__.logger
            if class_logger == 'silent':
                pass
            else:
                print_log(
                    msg=self.__class__.__get_key_warn_msg__(key),
                    logger=class_logger,
                    level=logging.WARN)
        val_check = self.__check_value__(key, val)
        if not val_check:
            raise ValueError(self.__class__.__get_value_error_msg__())
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
            val (Any):
                Value to the key.

        Returns:
            bool:
                True for matched, ortherwise False.
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
        """
        ret_bool = self.__check_value_type__(key, val) and\
            self.__check_value_shape__(key, val) and\
            self.__check_value_temporal__(key, val)
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
                try:
                    val_shape = val.shape
                except AttributeError:
                    # no shape attr
                    val_shape = []
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
    def temporal_len(self):
        return self.__temporal_len__

    @temporal_len.setter
    def temporal_len(self, value: int):
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

    @classmethod
    def __get_key_warn_msg__(cls, key: Any) -> str:
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
        error_message = \
            'An supported value doesn\'t ' +\
            'match definition.\n'
        suggestion_message = \
            'See error log for details.\n'
        return error_message + suggestion_message
