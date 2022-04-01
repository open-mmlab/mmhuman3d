import numpy as np
import pytest

from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_num
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.human_data_cache import (
    HumanDataCacheReader,
    HumanDataCacheWriter,
)

human_data_cache_path = 'tests/data/human_data/human_data_cache.npz'


def test_write():
    human_data = HumanData.new(key_strict=False)
    keypoint_num_hd = get_keypoint_num(convention='human_data')
    human_data['keypoints2d'] = np.ones(shape=(100, keypoint_num_hd, 3))
    human_data['keypoints2d_mask'] = np.ones(shape=(keypoint_num_hd, ))
    human_data['keypoints2d_convention'] = 'human_data'
    human_data['keypoints2d'][50, 50, :] *= 3
    keypoint_num_smplx = get_keypoint_num(convention='smplx')
    human_data['keypoints4d'] = np.ones(shape=(100, keypoint_num_smplx, 3))
    human_data['keypoints4d_mask'] = np.ones(shape=(keypoint_num_smplx, ))
    human_data['keypoints4d_mask'][0:10] *= 0
    human_data['keypoints4d_convention'] = 'smplx'
    human_data['config'] = 'config/example'
    human_data['image_path'] = [str(x) for x in range(100)]
    human_data['smpl'] = {
        'global_orient': np.ones(shape=(100, 3)),
        'betas': np.ones(shape=(100, 10)),
        'body_pose': np.ones(shape=(100, 21, 3)),
        'expression': [
            0,
        ],
    }
    human_data.compress_keypoints_by_mask()

    writer_kwargs, sliced_data = human_data.get_sliced_cache()
    writer = HumanDataCacheWriter(**writer_kwargs)
    writer.update_sliced_dict(sliced_data)
    writer.dump(human_data_cache_path, overwrite=True)
    with pytest.raises(FileExistsError):
        writer.dump(human_data_cache_path, overwrite=False)


def test_read():
    reader = HumanDataCacheReader(npz_path=human_data_cache_path)
    config = reader.get_non_sliced_data('config')
    assert config == 'config/example'
    slice_with_0 = reader.get_item(0)
    assert isinstance(slice_with_0, HumanData)
    slice_with_50 = reader.get_item(50)
    assert slice_with_50['image_path'][0] == '50'
    assert slice_with_50['keypoints2d'][0, 50, 1] == 3
    assert 'config' not in slice_with_50.keys()
    slice_with_config = reader.get_item(51, required_keys=['config'])
    assert 'config' in slice_with_config.keys()
    slice_compressed = reader.get_item(52)
    assert slice_compressed.get_raw_value('keypoints4d').shape[2] < 144
    slice_with_smpl = reader.get_item(52)
    assert slice_with_smpl['smpl']['body_pose'].shape[1:] == (21, 3)
    assert 'expression' not in slice_with_smpl['smpl']
    slice_with_smpl = reader.get_item(53, ['smpl'])
    assert slice_with_smpl['smpl']['global_orient'].shape[1] == 3
    assert slice_with_smpl['smpl']['expression'][0] == 0
