import logging
import os

import numpy as np
import pytest
import torch

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
    # set item correctly
    human_data = HumanData()
    sample_keypoints2d = np.zeros(shape=[3, 144, 3])
    human_data['keypoints2d'] = sample_keypoints2d
    # set item with mask
    human_data = HumanData()
    sample_keypoints2d = np.zeros(shape=[3, 144, 3])
    sample_keypoints2d_mask = np.zeros(shape=[144])
    sample_keypoints2d_mask[0:4] = 1
    human_data['keypoints2d_mask'] = sample_keypoints2d_mask
    human_data['keypoints2d'] = sample_keypoints2d
    assert shape_equal(human_data['keypoints2d'], sample_keypoints2d)
    non_zero_padding_kp2d = human_data.get_raw_value('keypoints2d')
    assert non_zero_padding_kp2d.shape[1] == 4
    human_data.set_raw_value('keypoints2d', non_zero_padding_kp2d)
    assert shape_equal(human_data['keypoints2d'], sample_keypoints2d)
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


def test_padding_and_slice():
    human_data_load_path = 'tests/data/human_data/human_data_00.npz'
    human_data = HumanData()
    human_data.load(human_data_load_path)
    assert human_data['keypoints2d'].shape[2] == 3
    # raw shape: 199, 18, 3
    raw_value = human_data.get_raw_value('keypoints2d')
    # test type Error
    with pytest.raises(AssertionError):
        human_data.get_value_in_shape('misc', [
            1,
        ])
    # test shape Error
    with pytest.raises(AssertionError):
        human_data.get_value_in_shape('keypoints2d', [200, 19])
    # test value Error
    with pytest.raises(ValueError):
        human_data.get_value_in_shape('keypoints2d', [0, 19, 4])
    # test padding
    padding_value = \
        human_data.get_value_in_shape('keypoints2d', [200, 19, 4])
    assert padding_value[199, 18, 3] == 0
    # test padding customed value
    padding_value = \
        human_data.get_value_in_shape('keypoints2d', [200, 19, 4], 1)
    assert padding_value[199, 18, 3] == 1
    # test slice
    slice_value = \
        human_data.get_value_in_shape('keypoints2d', [100, 10, 2])
    assert raw_value[0, 0, 0] == slice_value[0, 0, 0]
    # test padding + slice
    target_value = \
        human_data.get_value_in_shape('keypoints2d', [200, 10, 2])
    assert target_value[199, 9, 1] == 0
    assert raw_value[0, 0, 0] == target_value[0, 0, 0]
    # test default dim
    target_value = \
        human_data.get_value_in_shape('keypoints2d', [-1, 10, 2])
    assert raw_value.shape[0] == target_value.shape[0]


def test_temporal_slice():
    human_data_load_path = 'tests/data/human_data/human_data_00.npz'
    human_data = HumanData()
    human_data.load(human_data_load_path)
    assert human_data['keypoints2d'].shape[2] == 3
    # raw shape: 199, 18, 3
    raw_value = human_data.get_raw_value('keypoints2d')
    # slice with stop
    sliced_human_data = human_data.get_temporal_slice(1)
    assert sliced_human_data['keypoints2d'].shape[0] == 1
    assert \
        sliced_human_data.get_raw_value('keypoints2d')[0, 0, 0] == \
        raw_value[0, 0, 0]
    # slice with start and stop
    sliced_human_data = human_data.get_temporal_slice(1, 3)
    assert sliced_human_data['keypoints2d'].shape[0] == 2
    assert \
        sliced_human_data.get_raw_value('keypoints2d')[0, 0, 0] == \
        raw_value[1, 0, 0]
    # slice with start, stop and step
    sliced_human_data = human_data.get_temporal_slice(0, 5, 2)
    assert sliced_human_data['keypoints2d'].shape[0] == 3
    assert \
        sliced_human_data.get_raw_value('keypoints2d')[1, 0, 0] == \
        raw_value[2, 0, 0]


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


def test_to_device():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    default_device = torch.device(device_name)
    human_data_load_path = 'tests/data/human_data/human_data_00.npz'
    human_data = HumanData.fromfile(human_data_load_path)
    tensor_dict = human_data.to(default_device)
    assert tensor_dict['keypoints2d'].size(2) == 3
    # if cuda is available, test whether it is on gpu
    if 'cuda' in device_name:
        assert tensor_dict['keypoints2d'].get_device() == 0
        # default to cpu
        tensor_dict = human_data.to()
    assert tensor_dict['keypoints2d'].is_cuda is False


def shape_equal(ndarray_0, ndarray_1):
    shape_0 = np.asarray(ndarray_0.shape)
    shape_1 = np.asarray(ndarray_1.shape)
    if shape_0.ndim != shape_1.ndim:
        return False
    else:
        diff_value = np.abs(shape_0 - shape_1).sum()
        if diff_value == 0:
            return True
        else:
            return False
