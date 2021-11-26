import os

import pytest

from mmhuman3d.utils.path_utils import (
    Existence,
    check_input_path,
    check_path_existence,
    prepare_output_path,
)


def test_existence():
    empty_dir = 'test/data/path_utils/'
    os.makedirs(empty_dir, exist_ok=False)
    check_path_existence(empty_dir, 'file') == Existence.DirectoryExistEmpty
    os.removedirs(empty_dir)
    check_path_existence('mmhuman3d/core',
                         'file') == Existence.DirectoryExistNotEmpty
    check_path_existence('mmhuman3d/not_exist/',
                         'file') == Existence.DirectoryNotExist
    check_path_existence('mmhuman3d/not_exist.txt',
                         'auto') == Existence.FileNotExist


def test_prepare_output_path():
    prepare_output_path('tests/data/', overwrite=True)


def test_check_input_path():
    with pytest.raises(FileNotFoundError):
        check_input_path(
            'tests/data/human_data/human_data_00.npz',
            allowed_suffix=[
                '.pkl',
            ])
