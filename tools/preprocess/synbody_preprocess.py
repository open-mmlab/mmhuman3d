import argparse
import glob
import json
import os
import pdb
import time
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

# import mmcv
# from mmhuman3d.models.body_models.builder import build_body_model
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.data.data_structures.human_data import HumanData


def _get_imgname(v):
    rgb_folder = os.path.join(v, 'rgb')
    root_folder_id = v.split(os.path.sep).index(prefix)
    imglist = os.path.join(
        os.path.sep.join(v.split(os.path.sep)[root_folder_id:]), 'rgb')

    # image 1 is T-pose, don't use
    im = []
    images = [img for img in os.listdir(rgb_folder) if img.endswith('.jpeg')]
    for i in range(1, len(images)):
        imglist_tmp = os.path.join(imglist, f'{i:04d}.jpeg')
        im.append(imglist_tmp)

    # import pdb; pdb.set_trace()
    return im


def _get_exrname(v):
    rgb_folder = os.path.join(v, 'rgb')
    root_folder_id = v.split(os.path.sep).index(prefix)
    masklist = os.path.join(
        os.path.sep.join(v.split(os.path.sep)[root_folder_id:]), 'mask')

    # image 1 is T-pose, don't use
    images = [img for img in os.listdir(rgb_folder) if img.endswith('.jpeg')]
    exr = []
    for i in range(1, len(images)):
        masklist_tmp = os.path.join(masklist, f'{i:04d}.exr')
        exr.append(masklist_tmp)

    return exr


def _get_npzname(p, f_num):
    npz = []
    npz_tmp = p.split(os.path.sep)[-1]
    for _ in range(f_num):
        npz.append(npz_tmp)

    return npz


def _get_mask_conf(root, merged):
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

    root_folder_id = root.split(os.path.sep).index(prefix)

    conf = []
    for idx, mask_path in enumerate(
            merged['mask_path']):  # , desc='Frame kps conf'):
        exr_path = os.path.join(
            os.path.sep.join(root.split(os.path.sep)[:root_folder_id]),
            mask_path)

        # import pdb; pdb.set_trace()

        image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        json_path = os.path.join(
            os.path.sep.join(exr_path.split(os.path.sep)[:-2]),
            'seq_data.json')
        jsfile_tmp = json.load(open(json_path, 'r'))

        p_rgb = [0, 0, 0]
        keys = list(jsfile_tmp['Actors']['CharacterActors'].keys())

        found = False
        for key in keys:
            # synbody v0 applies
            if jsfile_tmp['Actors']['CharacterActors'][key][
                    'animation'] == merged['npz_name'][idx][:-4]:
                p_rgb = jsfile_tmp['Actors']['CharacterActors'][key][
                    'mask_rgb_value']
                found = True
        if not found:
            for key in keys:
                # synbody v1 applies
                if jsfile_tmp['Actors']['CharacterActors'][key][
                        'name_in_seq'] == merged['npz_name'][idx][:-4]:
                    p_rgb = jsfile_tmp['Actors']['CharacterActors'][key][
                        'mask_rgb_value']
                    break
        if p_rgb == [0, 0, 0]:
            raise ValueError(
                f'Cannot find info of {merged["npz_name"][idx][:-4]} in {json_path}'
            )

        kps2d = merged['keypoints2d'][idx]
        v = []

        for kp in kps2d:
            if (not 0 < kp[1] < 720) or (not 0 < kp[0] < 1280) or \
                    sum(image[int(kp[1]), int(kp[0])] * 255 - np.array(p_rgb)) > 3:
                v.append(0)
            else:
                v.append(1)
        conf.append(v)

    return conf


def process_npz(args):
    seq = args.seq
    # pdb.set_trace()
    root_folder_id = seq.split(os.path.sep).index(args.prefix)
    root_path = os.path.sep.join(seq.split(os.path.sep)[:root_folder_id])

    dataset_path = os.path.join(root_path, args.prefix)
    # pdb.set_trace()
    batch_name, place, seq_name = seq.split(os.path.sep)[root_folder_id + 1:]


    # batch name mapping
    batch_name_map = {
        '20230113': 'v1_0_train',
        '20230204': 'v1_0_train',
        '20230209': 'v1_0_train',
        '20230220': 'v1_0_train',
        '20230306_efh': 'v1_0_ehf',  # name change efh -> ehf
        '20230307_ehf': 'v1_0_ehf',
        '20230327_amass': 'v1_0_amass',
        '20230421_agora': 'v1_0_agora',
        '20230526_renew': 'v1_0_renew',
        '20230727': 'v1_1_train',
        '20240221': 'whac',
    }
    if not batch_name in batch_name_map.keys():
        return
    outpath = os.path.join(args.output_path, batch_name_map[batch_name],
                           batch_name, place, f'{seq_name}.npz')

    # assert not os.path.exists(outpath), f'{outpath} exist, skip!!'
    if os.path.exists(outpath):
        print(f'{outpath} exists, skip!!')
        return

    merged = {}
    # for key in ['image_path', 'mask_path', 'npz_name', 'meta', 'keypoints2d', 'keypoints3d', 'occlusion']:
    for key in [
        'image_path', 'mask_path', 'npz_name', 'meta', 'keypoints2d',
        'keypoints3d'
    ]:
        merged[key] = []
    merged['smpl'] = {}
    for key in ['transl', 'global_orient', 'betas', 'body_pose']:
        merged['smpl'][key] = []
    merged['smplx'] = {}
    for key in [
        'transl', 'global_orient', 'betas', 'body_pose', 'left_hand_pose',
        'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose',
        'expression'
    ]:
        merged['smplx'][key] = []

    try:
        imgname = _get_imgname(seq)
        exrname = _get_exrname(seq)
        valid_frame_number = len(imgname)
        # for p in tqdm(glob.glob(v + '/smpl_with_joints/*.npz'), desc='person'):

        # check if smpl / smplx annotation exists
        has_smpl, has_smplx = False, False
        smpl_f, smplx_f = '', ''
        seq_cont = os.listdir(seq)
        for fn in seq_cont:
            if fn.startswith('smpl_'):
                has_smpl = True
            if fn.startswith('smplx_'):
                has_smplx = True
        SMPL_FOLDER_NAMES = ['smpl_refit_withJoints_inCamSpace_noBUG',
                             'smpl_refit_withJoints_inCamSpace',
                             'smpl_withJoints_inCamSpace_noBUG',
                             'smpl_withJoints_inCamSpace']
        SMPLX_FOLDER_NAMES = ['smplx_withJoints_inCamSpace_noBUG',
                              'smplx_withJoints_inCamSpace']
        if has_smpl:
            for fn in SMPL_FOLDER_NAMES:
                if fn in seq_cont:
                    smpl_f = fn
                    break
            if len(smpl_f) == 0:
                has_smpl = False
        assert has_smpl
        if has_smplx:
            for fn in SMPLX_FOLDER_NAMES:
                if fn in seq_cont:
                    smplx_f = fn
                    break
            if len(smplx_f) == 0:
                has_smplx = False
        assert len(smplx_f) + len(smpl_f) > 0, \
            f'No smpl or smplx annotation found in {seq}'

        # parse smpl / smplx annotation
        smpl_ps, smplx_ps = [], []
        if has_smpl and len(smpl_f) > 0:
            smpl_ps = [p for p in os.listdir(os.path.join(seq, smpl_f))
                       if p.endswith('.npz')]
        if has_smplx and len(smplx_f) > 0:
            smplx_ps = [p for p in os.listdir(os.path.join(seq, smplx_f))
                        if p.endswith('.npz')]
        assert len(smplx_ps) > 0, 'SMPLX annotation not found!'
        if len(smpl_ps) > 0:
            assert len(smpl_ps) == len(smplx_ps), \
                'SMPL annotation but not match SMPLX!'

            for p in sorted(smpl_ps):
                npfile_tmp = np.load(os.path.join(seq, smpl_f, p), allow_pickle=True)

                for key in ['betas', 'global_orient', 'transl', 'body_pose']:
                    if key == 'betas' and len(
                            npfile_tmp['smpl'].item()['betas']) == 1:
                        betas = np.repeat(
                            npfile_tmp['smpl'].item()[key],
                            valid_frame_number,
                            axis=0)
                        merged['smpl']['betas'].append(betas)
                    else:
                        if len(npfile_tmp['smpl'].item()
                               [key]) == valid_frame_number:
                            merged['smpl'][key].append(
                                npfile_tmp['smpl'].item()[key])
                        else:
                            merged['smpl'][key].append(npfile_tmp['smpl'].item()
                                                       [key][1:valid_frame_number +
                                                               1])
        for p in sorted(smplx_ps):
            merged['image_path'] += imgname
            merged['mask_path'] += exrname
            merged['npz_name'] += _get_npzname(p, valid_frame_number)

            npfile_tmp = np.load(os.path.join(seq, smplx_f, p), allow_pickle=True)
            # pdb.set_trace()
            for _ in range(valid_frame_number):
                merged['meta'].append(npfile_tmp['meta'].item())
            merged['keypoints2d'].append(
                npfile_tmp['keypoints2d'][1:valid_frame_number + 1])
            merged['keypoints3d'].append(
                npfile_tmp['keypoints3d'][1:valid_frame_number + 1])
            # pdb.set_trace()
            for key in ['betas', 'global_orient', 'transl', 'body_pose',
                        'left_hand_pose', 'right_hand_pose', 'jaw_pose',
                        'leye_pose', 'reye_pose', 'expression']:
                if key == 'betas' and len(
                        npfile_tmp['smplx'].item()['betas']) == 1:
                    betas = np.repeat(
                        npfile_tmp['smplx'].item()[key],
                        valid_frame_number,
                        axis=0)
                    merged['smplx']['betas'].append(betas)
                else:
                    if len(npfile_tmp['smplx'].item()
                           [key]) == valid_frame_number:
                        merged['smplx'][key].append(
                            npfile_tmp['smplx'].item()[key])
                    else:
                        merged['smplx'][key].append(
                            npfile_tmp['smplx'].item()[key]
                            [1:valid_frame_number + 1])
        if has_smpl:
            for k in merged['smpl'].keys():
                merged['smpl'][k] = np.vstack(merged['smpl'][k])
        if has_smplx:
            for k in merged['smplx'].keys():
                merged['smplx'][k] = np.vstack(merged['smplx'][k])
            for k in ['left_hand_pose', 'right_hand_pose']:
                merged['smplx'][k] = merged['smplx'][k].reshape(-1, 15, 3)
            merged['smplx']['body_pose'] = merged['smplx']['body_pose'].reshape(
                -1, 21, 3)

        merged['keypoints3d'] = np.vstack(merged['keypoints3d'])
        merged['keypoints2d'] = np.vstack(merged['keypoints2d'])
        # merged['occlusion'] = np.vstack(merged['occlusion'])

        merged['conf'] = np.vstack(_get_mask_conf(dataset_path,
                                                  merged)).reshape([-1, 144, 1])

        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        np.savez(outpath, **merged)

        # pdb.set_trace()

    # except Exception as e:
    except FileNotFoundError as e:
        # failed.append(seq)
        # with open('log_synbody.json', 'w') as f:
        #     json.dump(failed, f)
        print(f'{batch_name}, {place}, {seq_name}', 'failed because of', e)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert datasets')

    parser.add_argument(
        '--seq', type=str, required=True, help='absolute path to the sequence')

    # parser.add_argument(
    #     '--root_path',
    #     type=str,
    #     required=True,
    #     help='the root path of original dataset')

    parser.add_argument(
        '--output_path',
        type=str,
        required=False,
        default='/mnt/lustre/weichen1/synbody_preprocess',
        help='the high level store folder of the preprocessed npz files')

    parser.add_argument(
        '--prefix',
        type=str,
        required=False,
        default='synbody',
        help='dataset folder name')

    # parser.add_argument(
    #     '--modes',
    #     type=str,
    #     nargs='+',
    #     required=False,
    #     default=[],
    #     help=f'Need to comply with supported modes specified in tools/convert_datasets.py')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    prefix = args.prefix
    process_npz(args)
