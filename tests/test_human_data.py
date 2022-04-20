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
    sample_keypoints2d_mask = np.ones(shape=[144])
    human_data['keypoints2d_mask'] = sample_keypoints2d_mask
    human_data['keypoints2d_convention'] = 'smplx'
    human_data['keypoints2d'] = sample_keypoints2d
    # set item without mask
    human_data = HumanData()
    sample_keypoints2d = np.zeros(shape=[3, 144, 3])
    human_data['keypoints2d_convention'] = 'smplx'
    human_data['keypoints2d'] = sample_keypoints2d
    # strict==True does not allow unsupported keys
    human_data.set_key_strict(True)
    with pytest.raises(KeyError):
        human_data['keypoints4d_mask'] = np.ones(shape=[
            144,
        ])
        human_data['keypoints4d_convention'] = 'smplx'
        human_data['keypoints4d'] = np.zeros(shape=[3, 144, 5])
    # strict==False allows unsupported keys
    human_data = HumanData.new(key_strict=False)
    human_data['keypoints4d_mask'] = np.ones(shape=[
        144,
    ])
    human_data['keypoints4d_convention'] = 'smplx'
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
    # test wrong value data len
    human_data = HumanData()
    human_data['bbox_xywh'] = np.zeros(shape=[2, 5])
    with pytest.raises(ValueError):
        human_data['keypoints2d_mask'] = np.ones(shape=[
            144,
        ])
        human_data['keypoints2d_convention'] = 'smplx'
        human_data['keypoints2d'] = np.zeros(shape=[3, 144, 3])
    # test everything is right
    human_data = HumanData()
    human_data['bbox_xywh'] = np.zeros(shape=[2, 5])
    human_data['keypoints2d_mask'] = np.ones(shape=[
        144,
    ])
    human_data['keypoints2d_convention'] = 'smplx'
    human_data['keypoints2d'] = np.zeros(shape=[2, 144, 3])


def test_compression():
    # set item with mask
    human_data = HumanData.new(key_strict=False)
    sample_keypoints2d = np.zeros(shape=[3, 144, 3])
    sample_keypoints2d_mask = np.zeros(shape=[144], dtype=np.uint8)
    sample_keypoints2d_mask[:5] += 1
    human_data['keypoints2d_mask'] = sample_keypoints2d_mask
    human_data['keypoints2d_convention'] = 'smplx'
    human_data['keypoints2d'] = sample_keypoints2d
    assert shape_equal(human_data['keypoints2d'], sample_keypoints2d)
    assert shape_equal(
        human_data.get_raw_value('keypoints2d'), sample_keypoints2d)
    human_data.set_raw_value('keypoints2d', sample_keypoints2d)
    assert shape_equal(human_data['keypoints2d'], sample_keypoints2d)
    # compress when mask is missing
    human_data.pop('keypoints2d_mask')
    with pytest.raises(KeyError):
        human_data.compress_keypoints_by_mask()
    # compress correctly
    assert human_data.check_keypoints_compressed() is False
    human_data['keypoints2d_mask'] = sample_keypoints2d_mask
    human_data.compress_keypoints_by_mask()
    assert human_data.check_keypoints_compressed() is True
    non_zero_padding_kp2d = human_data.get_raw_value('keypoints2d')
    assert shape_equal(human_data['keypoints2d'], sample_keypoints2d)
    assert non_zero_padding_kp2d.shape[1] < sample_keypoints2d.shape[1]
    # re-compress a compressed humandata
    with pytest.raises(AssertionError):
        human_data.compress_keypoints_by_mask()
    # modify mask when compressed
    human_data['keypoints2d_mask'] = np.ones([
        144,
    ])
    assert np.sum(human_data['keypoints2d_mask']) < 144
    # modify keypoints by set_raw_value
    human_data.set_raw_value('keypoints2d', non_zero_padding_kp2d)
    assert shape_equal(human_data['keypoints2d'], sample_keypoints2d)
    # modify keypoints by setitem
    human_data['keypoints2d'] = sample_keypoints2d
    assert shape_equal(human_data['keypoints2d'], sample_keypoints2d)
    # test set new key after compression
    with pytest.raises(ValueError):
        human_data['keypoints3d'] = np.zeros(shape=[3, 144, 4])
    # keypoints without mask is put in a compressed human_data
    with pytest.raises(KeyError):
        human_data.update({'keypoints3d': np.zeros(shape=[3, 144, 4])})
        human_data.decompress_keypoints()
    if 'keypoints3d' in human_data:
        human_data.pop('keypoints3d')
    human_data.decompress_keypoints()
    assert shape_equal(human_data['keypoints2d'], sample_keypoints2d)
    assert shape_equal(
        human_data.get_raw_value('keypoints2d'), sample_keypoints2d)
    # decompress a not compressed humandata
    with pytest.raises(AssertionError):
        human_data.decompress_keypoints()


def test_generate_mask_from_keypoints():

    human_data = HumanData.new(key_strict=False)
    keypoints2d = np.random.rand(3, 144, 3)
    keypoints3d = np.random.rand(3, 144, 4)

    # set confidence
    keypoints2d[:, :72, -1] = 0
    keypoints3d[:, 72:, -1] = 0
    human_data['keypoints2d'] = keypoints2d
    human_data['keypoints3d'] = keypoints3d

    # test all keys
    with pytest.raises(KeyError):
        human_data['keypoints2d_mask']
    with pytest.raises(KeyError):
        human_data['keypoints3d_mask']
    human_data.generate_mask_from_confidence()
    assert 'keypoints2d_mask' in human_data
    assert (human_data['keypoints2d_mask'][:72] == 0).all()
    assert (human_data['keypoints2d_mask'][72:] == 1).all()
    assert 'keypoints3d_mask' in human_data
    assert (human_data['keypoints3d_mask'][72:] == 0).all()
    assert (human_data['keypoints3d_mask'][:72] == 1).all()

    # test str keys
    human_data.generate_mask_from_confidence(keys='keypoints2d')

    # test list of str keys
    human_data.generate_mask_from_confidence(
        keys=['keypoints2d', 'keypoints3d'])

    # test compression with generated mask
    human_data.compress_keypoints_by_mask()
    assert human_data.check_keypoints_compressed() is True


def test_pop_unsupported_items():
    # directly pop them
    human_data = HumanData.new(key_strict=False)
    human_data['keypoints4d_mask'] = np.ones(shape=[
        144,
    ])
    human_data['keypoints4d_convention'] = 'smplx'
    human_data['keypoints4d'] = np.zeros(shape=[3, 144, 3])
    human_data.pop_unsupported_items()
    assert 'keypoints4d' not in human_data
    # pop when switching strict mode
    human_data = HumanData.new(key_strict=False)
    human_data['keypoints4d_mask'] = np.ones(shape=[
        144,
    ])
    human_data['keypoints4d_convention'] = 'smplx'
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


def test_slice():
    human_data_load_path = 'tests/data/human_data/human_data_00.npz'
    human_data = HumanData()
    human_data.load(human_data_load_path)
    assert human_data['keypoints2d'].shape[2] == 3
    # raw shape: 199, 18, 3
    raw_value = human_data.get_raw_value('keypoints2d')
    # slice with stop
    sliced_human_data = human_data.get_slice(1)
    assert sliced_human_data['keypoints2d'].shape[0] == 1
    assert \
        sliced_human_data.get_raw_value('keypoints2d')[0, 0, 0] == \
        raw_value[0, 0, 0]
    # slice with start and stop
    sliced_human_data = human_data.get_slice(1, 3)
    assert sliced_human_data['keypoints2d'].shape[0] == 2
    assert \
        sliced_human_data.get_raw_value('keypoints2d')[0, 0, 0] == \
        raw_value[1, 0, 0]
    # slice with start, stop and step
    sliced_human_data = human_data.get_slice(0, 5, 2)
    assert sliced_human_data['keypoints2d'].shape[0] == 3
    assert \
        sliced_human_data.get_raw_value('keypoints2d')[1, 0, 0] == \
        raw_value[2, 0, 0]
    # do not slice image_path when it has a wrong length
    human_data.__data_len__ = 199
    human_data['image_path'] = ['1.jpg', '2.jpg']
    sliced_human_data = human_data.get_slice(1)
    assert len(sliced_human_data['image_path']) == 2
    # slice image_path when it is correct
    image_list = [f'{index:04d}.jpg' for index in range(199)]
    human_data['image_path'] = image_list
    sliced_human_data = human_data.get_slice(0, 5, 2)
    assert len(sliced_human_data['image_path']) == 3
    # slice when there's a value without __len__ method
    human_data['some_id'] = 4
    human_data['misc'] = {'hd_id_plus_1': 5}
    sliced_human_data = human_data.get_slice(0, 5, 2)
    assert sliced_human_data['some_id'] == human_data['some_id']
    assert sliced_human_data['misc']['hd_id_plus_1'] == \
        human_data['misc']['hd_id_plus_1']


def test_missing_attr():
    dump_hd_path = 'tests/data/human_data/human_data_missing_len.npz'
    human_data = HumanData()
    human_data['smpl'] = {
        'body_pose': np.ones((1, 21, 3)),
        'transl': np.ones((1, 3)),
        'betas': np.ones((1, 10)),
    }
    # human_data.__delattr__('__data_len__')
    human_data.__data_len__ = -1
    human_data.dump(dump_hd_path)
    human_data = HumanData()
    human_data.load(dump_hd_path)
    assert human_data.data_len == 1


def test_load():
    human_data_load_path = 'tests/data/human_data/human_data_00.npz'
    human_data = HumanData()
    human_data.load(human_data_load_path)
    assert human_data['keypoints2d'].shape[2] == 3
    assert isinstance(human_data['misc'], dict)
    human_data['image_path'] = ['1.jpg', '2.jpg']
    human_data_with_image_path = \
        'tests/data/human_data/human_data_img.npz'
    human_data.dump(human_data_with_image_path)
    human_data = HumanData.fromfile(human_data_with_image_path)
    assert isinstance(human_data['image_path'], list)
    os.remove(human_data_with_image_path)


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
    # set item with mask
    human_data = HumanData()
    sample_keypoints2d = np.zeros(shape=[3, 144, 3])
    sample_keypoints2d_mask = np.zeros(shape=[144])
    sample_keypoints2d_mask[:4] = 1
    human_data['keypoints2d_mask'] = sample_keypoints2d_mask
    human_data['keypoints2d'] = sample_keypoints2d
    # 3 frames before dump
    assert human_data.data_len == 3

    human_data_dump_path = 'tests/data/human_data/human_data_00_dump.npz'
    if check_path_existence(human_data_dump_path,
                            'file') == Existence.FileExist:
        os.remove(human_data_dump_path)
    human_data.dump(human_data_dump_path, overwrite=True)
    assert check_path_existence(human_data_dump_path, 'file') == \
        Existence.FileExist
    human_data_load = HumanData()
    human_data_load.load(human_data_dump_path)
    # 3 frames after load
    assert human_data_load.data_len == 3

    # compatibility for old bbox
    old_version_dict = {
        'bbox_xywh': np.zeros(shape=(3, 4)),
        'image_path': None
    }
    human_data.update(old_version_dict)
    human_data.dump(human_data_dump_path, overwrite=True)
    human_data_load = HumanData()
    human_data_load.load(human_data_dump_path)
    # 3 frames after load
    assert human_data_load['bbox_xywh'].shape[1] == 5
    assert 'image_path' not in human_data_load

    # wrong file extension
    with pytest.raises(ValueError):
        human_data.dump(
            human_data_dump_path.replace('.npz', '.jpg'), overwrite=True)
    # file exists
    with pytest.raises(FileExistsError):
        human_data.dump(human_data_dump_path, overwrite=False)


def test_dump_by_pickle():
    # set item with mask
    human_data = HumanData()
    sample_keypoints2d = np.zeros(shape=[3, 144, 3])
    sample_keypoints2d_mask = np.zeros(shape=[144])
    sample_keypoints2d_mask[:4] = 1
    human_data['keypoints2d_mask'] = sample_keypoints2d_mask
    human_data['keypoints2d'] = sample_keypoints2d
    human_data.compress_keypoints_by_mask()
    # 3 frames before dump
    assert human_data.data_len == 3

    human_data_dump_path = 'tests/data/human_data/human_data_00_dump.pkl'
    if check_path_existence(human_data_dump_path,
                            'file') == Existence.FileExist:
        os.remove(human_data_dump_path)
    human_data.dump_by_pickle(human_data_dump_path, overwrite=True)
    assert check_path_existence(human_data_dump_path, 'file') == \
        Existence.FileExist
    human_data_load = HumanData()
    human_data_load.load_by_pickle(human_data_dump_path)
    # 3 frames after load
    assert human_data_load.data_len == 3

    # compatibility for old bbox
    old_version_dict = {
        'bbox_xywh': np.zeros(shape=(3, 4)),
        'image_path': None
    }
    human_data.update(old_version_dict)
    human_data.dump_by_pickle(human_data_dump_path, overwrite=True)
    human_data_load = HumanData()
    human_data_load.load_by_pickle(human_data_dump_path)

    # wrong file extension
    with pytest.raises(ValueError):
        human_data.dump_by_pickle(
            human_data_dump_path.replace('.pkl', '.jpg'), overwrite=True)
    # file exists
    with pytest.raises(FileExistsError):
        human_data.dump_by_pickle(human_data_dump_path, overwrite=False)


def test_log():
    HumanData.set_logger('silent')
    logger_human_data = HumanData.new(key_strict=False)
    logger_human_data['new_key_0'] = 1
    logger = logging.getLogger()
    HumanData.set_logger(logger)
    logger_human_data['new_key_1'] = 1


def test_to_device():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    default_device = torch.device(device_name)
    human_data_load_path = 'tests/data/human_data/human_data_00.npz'
    human_data = HumanData.fromfile(human_data_load_path)
    human_data.set_key_strict(False)
    human_data['tensor_value'] = torch.zeros((2, 2))
    tensor_dict = human_data.to(default_device)
    assert tensor_dict['keypoints2d'].size(2) == 3
    # if cuda is available, test whether it is on gpu
    if 'cuda' in device_name:
        assert tensor_dict['keypoints2d'].get_device() == 0
        # default to cpu
        tensor_dict = human_data.to()
    assert tensor_dict['keypoints2d'].is_cuda is False


def test_concat():
    human_data_0 = HumanData()
    human_data_0['image_path'] = ['path_0', 'path_1']
    human_data_0['keypoints2d'] = np.zeros(shape=(2, 190, 3))
    human_data_0['keypoints2d_convention'] = 'human_data'
    human_data_0['keypoints2d_mask'] = np.ones(shape=(190))
    # test list and np
    human_data_1 = HumanData()
    human_data_1['image_path'] = ['path_2']
    human_data_1['keypoints2d'] = np.ones(shape=(1, 190, 3))
    human_data_1['keypoints2d_convention'] = 'human_data'
    human_data_1['keypoints2d_mask'] = np.ones(shape=(190))
    cat_human_data = HumanData.concatenate(human_data_0, human_data_1)
    assert cat_human_data['keypoints2d'].shape[0] == 3
    assert\
        cat_human_data['keypoints2d'].shape[1:] ==\
        human_data_0['keypoints2d'].shape[1:]
    assert cat_human_data['image_path'][2] == \
        human_data_1['image_path'][0]
    assert cat_human_data['keypoints2d'][2, 0, 0] == \
        human_data_1['keypoints2d'][0, 0, 0]
    # test different mask
    human_data_1 = HumanData()
    human_data_1['image_path'] = ['path_2']
    human_data_1['keypoints2d'] = np.ones(shape=(1, 190, 3))
    human_data_1['keypoints2d_convention'] = 'human_data'
    human_data_1['keypoints2d_mask'] = np.ones(shape=(190))
    human_data_1['keypoints2d_mask'][0:10] = 0
    cat_human_data = HumanData.concatenate(human_data_0, human_data_1)
    assert cat_human_data['keypoints2d_mask'][9] == 0
    # test keys only mentioned once
    human_data_0['bbox_xywh'] = np.zeros((2, 5))
    human_data_1['keypoints3d'] = np.ones(shape=(1, 190, 4))
    human_data_1['keypoints3d_convention'] = 'human_data'
    human_data_1['keypoints3d_mask'] = np.ones(shape=(190))
    cat_human_data = HumanData.concatenate(human_data_0, human_data_1)
    assert 'keypoints2d' in cat_human_data
    assert 'keypoints3d' not in cat_human_data
    assert 'keypoints3d_1' in cat_human_data
    assert\
        cat_human_data['keypoints3d_1'].shape ==\
        human_data_1['keypoints3d'].shape
    # test different definition of the same key
    human_data_0['names'] = 'John Cena'
    human_data_1['names'] = 'John_Xina'
    cat_human_data = HumanData.concatenate(human_data_0, human_data_1)
    assert 'names_0' in cat_human_data
    assert cat_human_data['names_1'] == human_data_1['names']
    # test sub-dict by smpl
    human_data_0['smpl'] = {
        'body_pose': np.zeros((2, 21, 3)),
        'transl': np.zeros((2, 3)),
        'betas': np.zeros((2, 10)),
    }
    human_data_1 = HumanData()
    human_data_1['smpl'] = {
        'body_pose': np.ones((1, 21, 3)),
        'transl': np.ones((1, 3)),
        'betas': np.ones((1, 10)),
    }
    cat_human_data = HumanData.concatenate(human_data_0, human_data_1)
    assert cat_human_data['smpl']['body_pose'][2, 0, 0] == \
        human_data_1['smpl']['body_pose'][0, 0, 0]
    assert cat_human_data['smpl']['transl'][1, 0] == \
        human_data_0['smpl']['transl'][1, 0]
    assert cat_human_data['smpl']['betas'][2, 1] == \
        human_data_1['smpl']['betas'][0, 1]
    # test sub-keys only mentioned once
    human_data_0['smpl']['gender'] = 'male'
    human_data_0['smpl'].pop('betas')
    human_data_1['smpl']['expresssion'] = np.ones((1, 10))
    human_data_1['smpl'].pop('transl')
    cat_human_data = HumanData.concatenate(human_data_0, human_data_1)
    assert 'gender' in cat_human_data['smpl']
    assert 'expresssion' in cat_human_data['smpl']


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
