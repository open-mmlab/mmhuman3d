import os

import numpy as np

from mmhuman3d.data.preprocessors.agora_pre import agora_extract
from mmhuman3d.data.preprocessors.coco_pre import coco_extract
from mmhuman3d.data.preprocessors.coco_wholebody_pre import coco_wb_extract
from mmhuman3d.data.preprocessors.h36m_pre import h36m_extract
from mmhuman3d.data.preprocessors.lsp_extended_pre import lsp_extended_extract
from mmhuman3d.data.preprocessors.lsp_pre import lsp_extract
from mmhuman3d.data.preprocessors.mpi_inf_3dhp_pre import mpi_inf_3dhp_extract
from mmhuman3d.data.preprocessors.mpii_pre import mpii_extract
from mmhuman3d.data.preprocessors.pw3d_pre import pw3d_extract
from mmhuman3d.data.preprocessors.up3d_pre import up3d_extract


def test_preprocess():
    root_path = 'tests/data/dataset_sample'
    os.makedirs('/tmp/preprocessed_npzs', exist_ok=True)
    output_path = '/tmp/preprocessed_npzs'

    PW3D_ROOT = os.path.join(root_path, '3DPW')
    pw3d_extract(PW3D_ROOT, output_path, mode='train')
    pw3d_extract(PW3D_ROOT, output_path, mode='test')

    assert os.path.exists('/tmp/preprocessed_npzs/' + '3dpw_test.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + '3dpw_train.npz')

    H36M_ROOT = os.path.join(root_path, 'h36m')
    h36m_extract(H36M_ROOT, output_path, mode='train', protocol=1)
    h36m_extract(H36M_ROOT, output_path, mode='valid', protocol=1)
    h36m_extract(H36M_ROOT, output_path, mode='valid', protocol=2)

    assert os.path.exists('/tmp/preprocessed_npzs/' + 'h36m_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' +
                          'h36m_valid_protocol1.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' +
                          'h36m_valid_protocol2.npz')

    COCO_ROOT = os.path.join(root_path, 'coco')
    coco_extract(COCO_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'coco_2014_train.npz')

    MPI_INF_3DHP_ROOT = os.path.join(root_path, 'mpi_inf_3dhp')
    mpi_inf_3dhp_extract(MPI_INF_3DHP_ROOT, output_path, 'train')
    mpi_inf_3dhp_extract(MPI_INF_3DHP_ROOT, output_path, 'test')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'mpi_inf_3dhp_test.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'mpi_inf_3dhp_train.npz')

    MPII_ROOT = os.path.join(root_path, 'mpii')
    mpii_extract(MPII_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'mpii_train.npz')

    AGORA_ROOT = os.path.join(root_path, 'agora')
    agora_extract(AGORA_ROOT, output_path, 'train')
    agora_extract(AGORA_ROOT, output_path, 'validation')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'agora_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'agora_validation.npz')

    LSP_ORIGINAL_ROOT = os.path.join(root_path, 'lsp_dataset_original')
    lsp_extract(LSP_ORIGINAL_ROOT, output_path, 'train')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'lsp_train.npz')

    LSP_ROOT = os.path.join(root_path, 'lsp_dataset')
    lsp_extract(LSP_ROOT, output_path, 'test')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'lsp_test.npz')

    HR_LSPET_ROOT = os.path.join(root_path, 'hr-lspet')
    lsp_extended_extract(HR_LSPET_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'hr-lspet_train.npz')

    UP3D_ROOT = os.path.join(root_path, 'up-3d')
    up3d_extract(UP3D_ROOT, output_path, 'train')
    up3d_extract(UP3D_ROOT, output_path, 'test')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'up3d_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'up3d_test.npz')

    COCO_WHOLEBODY_ROOT = os.path.join(root_path, 'coco_wholebody')
    coco_wb_extract(COCO_WHOLEBODY_ROOT, output_path, 'train')
    coco_wb_extract(COCO_WHOLEBODY_ROOT, output_path, 'val')
    assert os.path.exists('/tmp/preprocessed_npzs/' +
                          'coco_wholebody_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'coco_wholebody_val.npz')


def test_preprocessed_npz():
    npz_folder = '/tmp/preprocessed_npzs'
    assert os.path.exists(npz_folder)
    all_keys = [
        'image_path', 'bbox_xywh', 'config', 'keypoints2d', 'keypoints3d',
        'smpl', 'smplx', 'meta', 'mask'
    ]

    for npf in os.listdir(npz_folder):
        npfile = np.load(os.path.join(npz_folder, npf), allow_pickle=True)
        assert 'image_path' in npfile
        N = npfile['image_path'].shape[0]

        for k in npfile.files:
            assert (k in all_keys)
            assert isinstance(npfile[k], np.ndarray)

            # check shape of every attributes
            if k == 'image_path':
                assert isinstance(npfile[k][0], np.str_)
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
                meta_keys = ['gender', 'kid', 'age', 'occlusion', 'ethnicity']
                meta_dict = npfile[k].item()
                for meta_key in meta_dict.keys():
                    assert meta_key in meta_keys
                    assert meta_dict['gender'].shape == (N, )

            elif k == 'mask':
                assert npfile[k].shape == (144, )
