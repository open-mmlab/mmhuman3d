import numpy as np
import pytest

from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.human_data_cache import (
    HumanDataCacheReader,
    HumanDataCacheWriter,
)

human_data_cache_path = 'tests/data/human_data/human_data_cache.npz'


def test_write():
    human_data = HumanData.new(key_strict=False)
    human_data['keypoints2d'] = np.ones(shape=(100, 199, 3))
    human_data['keypoints2d_mask'] = np.ones(shape=(199, ))
    human_data['keypoints2d_convention'] = 'human_data'
    human_data['keypoints2d'][50, 50, :] *= 3
    human_data['keypoints4d'] = np.ones(shape=(100, 144, 3))
    human_data['keypoints4d_mask'] = np.ones(shape=(144, ))
    human_data['keypoints4d_mask'][0:10] *= 0
    human_data['keypoints4d_convention'] = 'smplx'
    human_data['config'] = 'config/example'
    human_data['image_path'] = [x for x in range(100)]
    human_data.compress_keypoints_by_mask()

    writer_kwargs, sliced_data = human_data.get_sliced_cache()
    writer = HumanDataCacheWriter(**writer_kwargs)
    writer.update_sliced_dict(sliced_data)
    writer.dump(human_data_cache_path, overwrite=True)
    with pytest.raises(FileExistsError):
        writer.dump(human_data_cache_path, overwrite=False)


def test_read():
    reader = HumanDataCacheReader(npz_path=human_data_cache_path)
    slice_containing_0 = reader.get_item(0)
    assert isinstance(slice_containing_0, HumanData)
    slice_containing_50 = reader.get_item(50)
    assert slice_containing_50['image_path'][0] == 50
    assert slice_containing_50['keypoints2d'][0, 50, 1] == 3
    assert 'config' not in slice_containing_50.keys()
    slice_containing_config = reader.get_item(51, required_keys=['config'])
    assert 'config' in slice_containing_config.keys()
    slice_compressed = reader.get_item(52)
    assert slice_compressed.get_raw_value('keypoints4d').shape[2] < 144
