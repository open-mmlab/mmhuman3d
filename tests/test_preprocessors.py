import os

import numpy as np


def test_preprocessed_npz():
    npz_folder = './tests/data/preprocessed_datasets'
    assert os.path.exists(npz_folder)
    all_keys = [
        'image_path', 'bbox_xywh', 'config', 'keypoints2d', 'keypoints3d',
        'smpl', 'smplx', 'meta', 'mask'
    ]

    for npf in os.listdir(npz_folder):
        npfile = np.load(os.path.join(npz_folder, npf), allow_pickle=True)
        N = 1
        for k in npfile.files:
            assert (k in all_keys)
            assert isinstance(npfile[k], np.ndarray)

            # check shape of every attributes
            if k == 'image_path':
                assert isinstance(npfile[k][0], np.str_)
                assert npfile[k].shape == (N, )
            elif k == 'bbox_xywh':
                assert npfile[k].shape == (N, 4)

            elif k == 'config':
                assert npfile[k].shape == ()

            elif k == 'keypoints2d':
                assert npfile[k].shape == (N, 144, 3)

            elif k == 'keypoints3d':
                assert npfile[k].shape == (N, 144, 4)

            elif k == 'smpl':
                assert isinstance(npfile[k].item(), dict)
                smpl_keys = ['body_pose', 'global_orient', 'betas', 'transl']
                smpl_dict = npfile[k].item()
                for smpl_key in smpl_dict.keys():
                    assert smpl_key in smpl_keys
                    if smpl_key == 'body_pose':
                        assert smpl_dict[smpl_key].shape == (N, 23, 3)
                    elif smpl_key == 'global_orient':
                        assert smpl_dict[smpl_key].shape == (N, 3)
                    elif smpl_key == 'betas':
                        assert smpl_dict[smpl_key].shape == (N, 10)
                    elif smpl_key == 'transl':
                        assert smpl_dict[smpl_key].shape == (N, 3)

            elif k == 'smplx':
                smplx_keys = [
                    'body_pose', 'global_orient', 'betas', 'transl',
                    'left_hand_pose', 'right_hand_pose', 'expression',
                    'leye_pose', 'reye_pose', 'jaw_pose'
                ]
                smplx_dict = npfile[k].item()
                for smplx_key in smplx_dict.keys():
                    assert smplx_key in smplx_keys
                    if smplx_key == 'body_pose':
                        assert smplx_dict[smplx_key].shape == (N, 21, 3)
                    elif smplx_key == 'global_orient':
                        assert smplx_dict[smplx_key].shape == (N, 3)
                    elif smplx_key == 'betas':
                        assert smplx_dict[smplx_key].shape == (N, 10)
                    elif smplx_key == 'transl':
                        assert smplx_dict[smplx_key].shape == (N, 3)
                    elif smplx_key == 'left_hand_pose':
                        assert smplx_dict[smplx_key].shape == (N, 15, 3)
                    elif smplx_key == 'right_hand_pose':
                        assert smplx_dict[smplx_key].shape == (N, 15, 3)
                    elif smplx_key == 'expression':
                        assert smplx_dict[smplx_key].shape == (N, 10)
                    elif smplx_key == 'leye_pose':
                        assert smplx_dict[smplx_key].shape == (N, 3)
                    elif smplx_key == 'reye_pose':
                        assert smplx_dict[smplx_key].shape == (N, 3)
                    elif smplx_key == 'jaw_pose':
                        assert smplx_dict[smplx_key].shape == (N, 3)

            elif k == 'meta':
                meta_keys = ['gender']
                meta_dict = npfile[k].item()
                for meta_key in meta_dict.keys():
                    assert meta_key in meta_keys
                    assert meta_dict['gender'].shape == (N, )

            elif k == 'mask':
                assert npfile[k].shape == (144, )
