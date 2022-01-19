import numpy as np
import pytest

from mmhuman3d.data.datasets.pipelines import LoadImageFromFile

test_image_path = 'tests/data/dataset_sample/3DPW/imageFiles/' \
                  'courtyard_arguing_00/image_00000.jpg'
test_smc_path = 'tests/data/dataset_sample/humman/p000003_a000014_tiny.smc'


def test_load_image_from_file():
    results = {'img_prefix': None, 'image_path': test_image_path}

    pipeline = LoadImageFromFile()
    results = pipeline(results)

    assert results['filename'] == results['ori_filename'] == test_image_path
    assert isinstance(results['img'], np.ndarray)
    assert results['img_shape'] == results['ori_shape'] == (1920, 1080)
    assert isinstance(results['img_norm_cfg'], dict)


def test_load_image_from_file_smc():
    results = {'img_prefix': None, 'image_path': test_smc_path}

    pipeline = LoadImageFromFile()

    with pytest.raises(AssertionError):
        results = pipeline(results)

    results['image_id'] = ('Kinect', 0, 0)
    results = pipeline(results)

    assert results['filename'] == results['ori_filename'] == test_smc_path
    assert isinstance(results['img'], np.ndarray)
    assert results['img_shape'] == results['ori_shape'] == (1080, 1920)
    assert isinstance(results['img_norm_cfg'], dict)

    results['image_id'] = ('iPhone', 0, 0)
    results = pipeline(results)

    assert results['filename'] == results['ori_filename'] == test_smc_path
    assert isinstance(results['img'], np.ndarray)
    assert results['img_shape'] == results['ori_shape'] == (1920, 1440)
    assert isinstance(results['img_norm_cfg'], dict)
