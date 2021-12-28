import os
import os.path as osp

import numpy as np

from mmhuman3d.data.data_converters import build_data_converter


def test_preprocess():
    root_path = 'tests/data/dataset_sample'
    output_path = '/tmp/preprocessed_npzs'
    os.makedirs(output_path, exist_ok=True)

    PW3D_ROOT = osp.join(root_path, 'pw3d')
    cfg = dict(type='Pw3dConverter', modes=['train', 'test'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(PW3D_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'pw3d_test.npz'))
    assert osp.exists(osp.join(output_path, 'pw3d_train.npz'))

    H36M_ROOT = osp.join(root_path, 'h36m')
    cfg = dict(
        type='H36mConverter',
        modes=['train', 'valid'],
        protocol=1,
        mosh_dir='tests/data/dataset_sample/h36m_mosh')
    data_converter = build_data_converter(cfg)
    data_converter.convert(H36M_ROOT, output_path)
    cfg = dict(type='H36mConverter', modes=['valid'], protocol=2)
    data_converter = build_data_converter(cfg)
    data_converter.convert(H36M_ROOT, output_path)

    assert osp.exists(osp.join(output_path, 'h36m_train.npz'))
    assert osp.exists(osp.join(output_path, 'h36m_valid_protocol1.npz'))
    assert osp.exists(osp.join(output_path, 'h36m_valid_protocol2.npz'))

    COCO_ROOT = osp.join(root_path, 'coco')
    cfg = dict(type='CocoConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert(COCO_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'coco_2014_train.npz'))

    MPI_INF_3DHP_ROOT = osp.join(root_path, 'mpi_inf_3dhp')
    cfg = dict(type='MpiInf3dhpConverter', modes=['train', 'test'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(MPI_INF_3DHP_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'mpi_inf_3dhp_test.npz'))
    assert osp.exists(osp.join(output_path, 'mpi_inf_3dhp_train.npz'))

    MPII_ROOT = osp.join(root_path, 'mpii')
    cfg = dict(type='MpiiConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert(MPII_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'mpii_train.npz'))

    PENN_ACTION_ROOT = osp.join(root_path, 'Penn_Action')
    cfg = dict(type='PennActionConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert(PENN_ACTION_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'penn_action_train.npz'))

    AGORA_ROOT = osp.join(root_path, 'agora')
    cfg = dict(
        type='AgoraConverter', modes=['train', 'validation'], fit='smplx')
    data_converter = build_data_converter(cfg)
    data_converter.convert(AGORA_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'agora_train_smplx.npz'))
    assert osp.exists(osp.join(output_path, 'agora_validation_smplx.npz'))

    LSP_ORIGINAL_ROOT = osp.join(root_path, 'lsp_dataset_original')
    cfg = dict(type='LspConverter', modes=['train'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(LSP_ORIGINAL_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'lsp_train.npz'))

    LSP_ROOT = osp.join(root_path, 'lsp_dataset')
    cfg = dict(type='LspConverter', modes=['test'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(LSP_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'lsp_test.npz'))

    HR_LSPET_ROOT = osp.join(root_path, 'hr-lspet')
    cfg = dict(type='LspExtendedConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert(HR_LSPET_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'lspet_train.npz'))

    UP3D_ROOT = osp.join(root_path, 'up-3d')
    cfg = dict(type='Up3dConverter', modes=['trainval', 'test'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(UP3D_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'up3d_trainval.npz'))
    assert osp.exists(osp.join(output_path, 'up3d_test.npz'))

    COCO_WHOLEBODY_ROOT = osp.join(root_path, 'coco_wholebody')
    cfg = dict(type='CocoWholebodyConverter', modes=['train', 'val'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(COCO_WHOLEBODY_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'coco_wholebody_train.npz'))
    assert osp.exists(osp.join(output_path, 'coco_wholebody_val.npz'))

    AMASS_ROOT = osp.join(root_path, 'AMASS_file')
    cfg = dict(type='AmassConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert(AMASS_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'amass.npz'))

    POSETRACK_ROOT = osp.join(root_path, 'PoseTrack/data')
    cfg = dict(type='PosetrackConverter', modes=['train', 'val'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(POSETRACK_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'posetrack_train.npz'))
    assert osp.exists(osp.join(output_path, 'posetrack_val.npz'))

    EFT_ROOT = os.path.join(root_path, 'eft')
    cfg = dict(
        type='EftConverter', modes=['coco_all', 'coco_part', 'mpii', 'lspet'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(EFT_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'eft_coco_all.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'eft_coco_part.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'eft_mpii.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'eft_lspet.npz')

    CROWDPOSE_ROOT = os.path.join(root_path, 'Crowdpose')
    cfg = dict(
        type='CrowdposeConverter', modes=['train', 'val', 'test', 'trainval'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(CROWDPOSE_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'crowdpose_val.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'crowdpose_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'crowdpose_test.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'crowdpose_trainval.npz')

    SURREAL_ROOT = os.path.join(root_path, 'SURREAL/cmu')
    cfg = dict(type='SurrealConverter', modes=['train', 'val', 'test'], run=0)
    data_converter = build_data_converter(cfg)
    data_converter.convert(SURREAL_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'surreal_val_run0.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'surreal_train_run0.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'surreal_test_run0.npz')

    HYBRIK_ROOT = os.path.join(root_path, 'hybrik_data')
    cfg = dict(type='Pw3dHybrIKConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert(HYBRIK_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'hybrik_pw3d_test.npz')

    cfg = dict(type='H36mHybrIKConverter', modes=['train', 'test'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(HYBRIK_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'hybrik_h36m_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' +
                          'hybrik_h36m_valid_protocol2.npz')

    cfg = dict(type='MpiInf3dhpHybrIKConverter', modes=['train', 'test'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(HYBRIK_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' +
                          'hybrik_mpi_inf_3dhp_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' +
                          'hybrik_mpi_inf_3dhp_test.npz')

    COCO_2017_ROOT = os.path.join(root_path, 'coco_2017')
    cfg = dict(type='CocoHybrIKConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert(COCO_2017_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' +
                          'hybrik_coco_2017_train.npz')

    VIBE_ROOT = os.path.join(root_path, 'vibe_data')
    cfg = dict(type='InstaVibeConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert(VIBE_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'insta_variety.npz')
    cfg = dict(
        type='VibeConverter',
        modes=['mpi_inf_3dhp', 'pw3d'],
        pretrained_ckpt=None)
    data_converter = build_data_converter(cfg)
    data_converter.convert(VIBE_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' +
                          'vibe_mpi_inf_3dhp_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'vibe_pw3d_test.npz')

    SPIN_ROOT = os.path.join(root_path, 'spin_data')
    cfg = dict(
        type='SpinConverter',
        modes=['coco_2014', 'lsp', 'mpii', 'mpi_inf_3dhp', 'lspet'])
    data_converter = build_data_converter(cfg)
    data_converter.convert(SPIN_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/' +
                          'spin_coco_2014_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'spin_lsp_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' +
                          'spin_mpi_inf_3dhp_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'spin_mpii_train.npz')
    assert os.path.exists('/tmp/preprocessed_npzs/' + 'spin_lspet_train.npz')
    cfg = dict(
        type='H36mSpinConverter',
        modes=['train'],
        mosh_dir='tests/data/dataset_sample/h36m_mosh')
    data_converter = build_data_converter(cfg)
    data_converter.convert(H36M_ROOT, output_path)
    assert osp.exists(osp.join(output_path, 'spin_h36m_train.npz'))

    GTA_HUMAN_ROOT = os.path.join(root_path, 'gta_human_data')
    cfg = dict(type='GTAHumanConverter')
    data_converter = build_data_converter(cfg)
    data_converter.convert(GTA_HUMAN_ROOT, output_path)
    assert os.path.exists('/tmp/preprocessed_npzs/gta_human.npz')


def test_preprocessed_npz():
    npz_folder = '/tmp/preprocessed_npzs'
    assert osp.exists(npz_folder)
    all_keys = [
        'image_path', 'bbox_xywh', 'config', 'keypoints2d', 'keypoints3d',
        'smpl', 'smplx', 'smplh', 'meta', 'keypoints2d_mask', 'video_path',
        'frame_idx', 'keypoints3d_mask', 'cam_param', 'image_height',
        'image_width', 'root_cam', 'depth_factor', 'keypoints3d17_cam_mask',
        'keypoints3d_cam_mask', 'keypoints3d17_mask',
        'keypoints3d17_relative_mask', 'keypoints3d_relative',
        'keypoints3d17_cam', 'keypoints3d17', 'keypoints3d17_relative',
        'keypoints3d_cam', 'keypoints3d_relative_mask', 'phi', 'phi_weight',
        'features', 'has_smpl', 'keypoints2d_gta', 'keypoints3d_gta',
        'keypoints2d_gta_mask', 'keypoints3d_gta_mask'
    ]

    for npf in os.listdir(npz_folder):
        npfile = np.load(osp.join(npz_folder, npf), allow_pickle=True)
        assert 'image_path' or 'video_path' in npfile
        if 'image_path' in npfile:
            N = npfile['image_path'].shape[0]
        else:
            assert 'frame_idx' in npfile
            N = npfile['video_path'].shape[0]

        for k in npfile.files:
            if not (k.startswith('__') and k.endswith('__')):
                assert (k in all_keys)
            assert isinstance(npfile[k], np.ndarray)

            # check shape of every attributes
            if k == 'image_path':
                assert isinstance(npfile[k][0], np.str_)

            elif k == 'video_path':
                assert isinstance(npfile[k][0], np.str_)

            elif k == 'frame_idx':
                assert npfile[k].shape == (N, )

            elif k == 'bbox_xywh':
                assert npfile[k].shape == (N, 5)

            elif k == 'config':
                assert npfile[k].shape == ()

            elif k == 'keypoints2d':
                N_keypoints = npfile[k].shape[1]
                assert npfile[k].shape == (N, N_keypoints, 3)

            elif k == 'keypoints3d':
                N_keypoints = npfile[k].shape[1]
                assert npfile[k].shape == (N, N_keypoints, 4)

            elif k == 'smpl':
                assert isinstance(npfile[k].item(), dict)
                smpl_keys = [
                    'body_pose', 'global_orient', 'betas', 'transl', 'thetas'
                ]
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
                    elif smpl_key == 'thetas':
                        assert smpl_dict[smpl_key].shape == (N, 24, 3)

            elif k == 'smplh':
                smplh_keys = [
                    'body_pose', 'global_orient', 'betas', 'transl',
                    'left_hand_pose', 'right_hand_pose'
                ]
                smplh_dict = npfile[k].item()
                for smplh_key in smplh_dict.keys():
                    assert smplh_key in smplh_keys
                    if smplh_key == 'body_pose':
                        assert smplh_dict[smplh_key].shape == (N, 21, 3)
                    elif smplh_key == 'global_orient':
                        assert smplh_dict[smplh_key].shape == (N, 3)
                    elif smplh_key == 'betas':
                        assert smplh_dict[smplh_key].shape == (N, 10)
                    elif smplh_key == 'transl':
                        assert smplh_dict[smplh_key].shape == (N, 3)
                    elif smplh_key == 'left_hand_pose':
                        assert smplh_dict[smplh_key].shape == (N, 15, 3)
                    elif smplh_key == 'right_hand_pose':
                        assert smplh_dict[smplh_key].shape == (N, 15, 3)

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

            elif 'mask' in k:
                assert npfile[k].shape == (190, )

            elif 'keypoints3d' in k and 'mask' not in k:
                N_keypoints = npfile[k].shape[1]
                assert npfile[k].shape == (N, N_keypoints, 4)

            elif k == 'image_height' or k == 'image_width':
                assert isinstance(npfile[k][0], np.int64)

            elif k == 'root_cam':
                assert npfile[k].shape == (N, 3)

            elif k == 'depth_factor':
                assert isinstance(npfile[k][0], np.float64)

            elif k == 'phi' or k == 'phi_weight':
                assert npfile[k].shape == (N, 23, 2)

            elif k == 'features':
                assert npfile[k].shape == (N, 2048)
