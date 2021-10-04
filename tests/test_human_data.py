import logging
import os

import numpy as np
import pytest

from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.utils.path_utils import Existence, check_path_existence


def test_new():
    human_data = HumanData()
    # no key shall exist in the empty HumanData
    assert len(human_data) == 0
    human_data = HumanData.new(source_dict=None, key_strict=True)
    assert len(human_data) == 0
    human_data = HumanData.new(source_dict=None, key_strict=False)
    assert len(human_data) == 0


def test_set():
    human_data = HumanData()
    human_data['keypoints2d'] = np.zeros(shape=[3, 144, 3])
    # strict==True does not allow unsupported keys
    human_data.set_key_strict(True)
    with pytest.raises(KeyError):
        human_data['keypoints4d'] = np.zeros(shape=[3, 144, 5])
    # strict==False allows unsupported keys
    human_data = HumanData.new(key_strict=False)
    human_data['keypoints4d'] = np.zeros(shape=[3, 144, 3])
    # test wrong value type
    with pytest.raises(ValueError):
        human_data = HumanData()
        human_data['image_path'] = 2
    # test wrong value shape
    with pytest.raises(ValueError):
        human_data = HumanData()
        human_data['bbox_xywh'] = np.zeros(shape=[2, 4])
    # test wrong value with no shape attr
    with pytest.raises(AttributeError):
        human_data = HumanData()
        bbox_np = np.zeros(shape=[2, 4])
        delattr(bbox_np, 'shape')
        human_data['bbox_xywh'] = bbox_np
    # test wrong value shape dim
    with pytest.raises(ValueError):
        human_data = HumanData()
        bbox_np = np.zeros(shape=[
            2,
        ])
        human_data['bbox_xywh'] = bbox_np
    # test wrong value temporal len
    human_data = HumanData()
    human_data['bbox_xywh'] = np.zeros(shape=[2, 5])
    with pytest.raises(ValueError):
        human_data['keypoints2d'] = np.zeros(shape=[3, 144, 3])
    # test everything is right
    human_data = HumanData()
    human_data['bbox_xywh'] = np.zeros(shape=[2, 5])
    human_data['keypoints2d'] = np.zeros(shape=[2, 144, 3])


def test_pop_unsupported_items():
    # directly pop them
    human_data = HumanData.new(key_strict=False)
    human_data['keypoints4d'] = np.zeros(shape=[3, 144, 3])
    human_data.pop_unsupported_items()
    assert 'keypoints4d' not in human_data
    # pop when switching strict mode
    human_data = HumanData.new(key_strict=False)
    human_data['keypoints4d'] = np.zeros(shape=[3, 144, 3])
    human_data.set_key_strict(True)
    assert 'keypoints4d' not in human_data


def test_load():
    human_data_load_path = 'tests/data/human_data/human_data_00.npz'
    human_data = HumanData()
    human_data.load(human_data_load_path)
    assert human_data['keypoints2d'].shape[2] == 3
    assert isinstance(human_data['misc'], dict)


def test_construct_from_dict():
    # strict==True does not allow unsupported keys
    dict_with_keypoints2d = {'keypoints2d': np.zeros(shape=[3, 144, 3])}
    human_data = HumanData(dict_with_keypoints2d)
    assert human_data['keypoints2d'].shape[2] == 3
    # strict==True does not allow unsupported keys
    human_data = HumanData.new(
        source_dict=dict_with_keypoints2d, key_strict=False)
    assert human_data['keypoints2d'].shape[2] == 3
    # strict==False allows unsupported keys
    dict_with_keypoints4d = {'keypoints4d': np.zeros(shape=[3, 144, 5])}
    human_data = HumanData.new(
        source_dict=dict_with_keypoints4d, key_strict=False)
    assert human_data['keypoints4d'].shape[2] == 5


def test_construct_from_file():
    human_data_load_path = 'tests/data/human_data/human_data_00.npz'
    human_data = HumanData.fromfile(human_data_load_path)
    assert human_data['keypoints2d'].shape[2] == 3


def test_dump():
    human_data_load_path = 'tests/data/human_data/human_data_00.npz'
    human_data = HumanData()
    human_data.load(human_data_load_path)
    human_data_dump_path = 'tests/data/human_data/human_data_00_dump.npz'
    if check_path_existence(human_data_dump_path,
                            'file') == Existence.FileExist:
        os.remove(human_data_dump_path)
    human_data.dump(human_data_dump_path, overwrite=True)
    assert check_path_existence(human_data_dump_path, 'file') == \
        Existence.FileExist
    # wrong file extension
    with pytest.raises(ValueError):
        human_data.dump(
            human_data_dump_path.replace('.npz', '.jpg'), overwrite=True)
    # file exists
    with pytest.raises(FileExistsError):
        human_data.dump(human_data_dump_path, overwrite=False)


def test_log():
    HumanData.set_logger('silent')
    test_set()
    logger = logging.getLogger()
    HumanData.set_logger(logger)
    test_set()
