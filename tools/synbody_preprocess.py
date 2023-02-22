import os
from typing import List

import argparse
import time
import numpy as np
import json
import cv2
import glob
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
# import mmcv
# from mmhuman3d.models.body_models.builder import build_body_model
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idxs_by_part

import pdb

def _get_imgname(v):
    rgb_folder = os.path.join(v, 'rgb')
    root_folder_id = v.split('/').index(prefix)
    imglist = os.path.join('/'.join(v.split('/')[root_folder_id:]), 'rgb')

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
    root_folder_id = v.split('/').index(prefix)
    masklist = os.path.join('/'.join(v.split('/')[root_folder_id:]), 'mask')

    # image 1 is T-pose, don't use
    images = [img for img in os.listdir(rgb_folder) if img.endswith('.jpeg')]
    exr = []
    for i in range(1, len(images)):
        masklist_tmp = os.path.join(masklist, f'{i:04d}.exr')
        exr.append(masklist_tmp)

    return exr

def _get_npzname(p, f_num):
    
    npz = []
    npz_tmp = p.split('/')[-1]
    for _ in range(f_num):
        npz.append(npz_tmp)

    return npz


def _get_mask_conf(root, merged):
    
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

    root_folder_id = root.split('/').index(prefix)

    conf = []
    for idx, mask_path in enumerate(merged['mask_path']): # , desc='Frame kps conf'):
        exr_path = os.path.join('/'.join(root.split('/')[:root_folder_id]), mask_path)

        # import pdb; pdb.set_trace()

        image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        json_path = os.path.join('/'.join(exr_path.split('/')[:-2]), 'seq_data.json')
        jsfile_tmp = json.load(open(json_path, 'r'))

        p_rgb = [0, 0, 0]
        keys = list(jsfile_tmp['Actors']['CharacterActors'].keys())
        for key in keys:
            if jsfile_tmp['Actors']['CharacterActors'][key]['animation'] == merged['npz_name'][idx][:-4]:
                p_rgb = jsfile_tmp['Actors']['CharacterActors'][key]['mask_rgb_value']
                break
        if p_rgb == [0, 0, 0]:
            raise ValueError(f'Cannot find info of {merged["npz_name"][idx][:-4]} in {json_path}')

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


# def process_npz(args):
#     # root_path is where the npz files stored. Should ends with 'synbody' or 'Synbody'
#     # if not os.path.basename(root_path).endswith('synbody'):
#     #     root_path = os.path.join(root_path, 'synbody')
#     # ple = [p for p in ple if '.' not in p]
#     dataset_path = os.path.join(args.root_path, args.prefix)
#     batch_paths = [os.path.join(args.root_path, p) for p in os.listdir(dataset_path)]

#     seqs_targeted = glob.glob(os.path.join(args.root_path, args.prefix, '*', '*', 'LS*'))
#     print(f'There are {len(batch_paths)} batches and {len(seqs_targeted)} sequences')

#     # print(ple)
#     os.makedirs(args.output_path, exist_ok=True)
#     # failed = []

#     for seq in tqdm(seqs_targeted, desc='sequences'):
        
#         root_folder_id = seq.split('/').index('synbody')
#         batch_name, place, seq_name = seq.split('/')[root_folder_id+1:]

#         merged = {}
#         for key in ['image_path', 'mask_path', 'npz_name', 'meta', 'keypoints2d', 'keypoints3d']:
#             merged[key] = []
#         merged['smpl'] = {}
#         for key in ['transl', 'global_orient', 'betas', 'body_pose']:
#             merged['smpl'][key] = []
#         merged['smplx'] = {}
#         for key in ['transl', 'global_orient', 'betas', 'body_pose', 
#                     'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
#             merged['smplx'][key] = []

#         try:
#             imgname = _get_imgname(seq)
#             exrname = _get_exrname(seq)
#             valid_frame_number = len(imgname)
#             # for p in tqdm(glob.glob(v + '/smpl_with_joints/*.npz'), desc='person'):
#             ps = [p for p in os.listdir(os.path.join(seq, 'smpl_withJoints_inCamSpace')) if p.endswith('.npz')]

#             for p in sorted(ps):
#                 npfile_tmp = np.load(os.path.join(seq, 'smpl_withJoints_inCamSpace', p), allow_pickle=True)
#                 merged['image_path'] += imgname
#                 merged['mask_path'] += exrname
#                 merged['npz_name'] += _get_npzname(p, valid_frame_number)

#                 for _ in range(valid_frame_number):
#                     merged['meta'].append(npfile_tmp['meta'])

#                 for key in ['betas', 'global_orient', 'transl', 'body_pose']:
#                     if key == 'betas' and len(npfile_tmp['smpl'].item()['betas']) == 1:
#                         betas = np.repeat(npfile_tmp['smpl'].item()[key], valid_frame_number, axis=0)
#                         merged['smpl']['betas'].append(betas)
#                     else:
#                         if len(npfile_tmp['smpl'].item()[key]) == valid_frame_number:
#                             merged['smpl'][key].append(npfile_tmp['smpl'].item()[key])
#                         else:
#                             merged['smpl'][key].append(npfile_tmp['smpl'].item()[key][1:valid_frame_number+1])

#                 npfile_tmp = np.load(os.path.join(seq, 'smplx_withJoints_inCamSpace', p), allow_pickle=True)
#                 merged['keypoints2d'].append(npfile_tmp['keypoints2d'][1:valid_frame_number+1])
#                 merged['keypoints3d'].append(npfile_tmp['keypoints3d'][1:valid_frame_number+1])

#                 for key in ['betas', 'global_orient', 'transl', 'body_pose', \
#                             'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
#                     if key == 'betas' and len(npfile_tmp['smplx'].item()['betas']) == 1:
#                         betas = np.repeat(npfile_tmp['smplx'].item()[key], valid_frame_number, axis=0)
#                         merged['smplx']['betas'].append(betas)
#                     else:
#                         if len(npfile_tmp['smplx'].item()[key]) == valid_frame_number:
#                             merged['smplx'][key].append(npfile_tmp['smplx'].item()[key])
#                         else:
#                             merged['smplx'][key].append(npfile_tmp['smplx'].item()[key][1:valid_frame_number+1])

#             for k in merged['smpl'].keys():
#                 merged['smpl'][k] = np.vstack(merged['smpl'][k])
#             for k in merged['smplx'].keys():
#                 merged['smplx'][k] = np.vstack(merged['smplx'][k])
#             for k in ['left_hand_pose', 'right_hand_pose']:
#                 merged['smplx'][k] = merged['smplx'][k].reshape(-1, 15, 3)
#             merged['smplx']['body_pose'] = merged['smplx']['body_pose'].reshape(-1, 21, 3)

#             merged['keypoints3d'] = np.vstack(merged['keypoints3d'])
#             merged['keypoints2d'] = np.vstack(merged['keypoints2d'])

#             merged['conf'] = np.vstack(_get_mask_conf(dataset_path, merged)).reshape(-1, 144, 1)

#             os.makedirs(os.path.join(args.output_path, batch_name, place), exist_ok=True)
#             outpath = os.path.join(args.output_path, batch_name, place, f'{seq_name}.npz')
#             np.savez(outpath, **merged)
                
#             # pdb.set_trace()
                
#         # except Exception as e:
#         except FileNotFoundError as e:
#             # failed.append(seq)
#             # with open('log_synbody.json', 'w') as f:
#             #     json.dump(failed, f)
#             print(f'{batch_name}, {place}, {seq_name}', 'failed because of', e)

def process_npz(args):

    seq = args.seq
    root_folder_id = seq.split('/').index(args.prefix)
    root_path = '/'.join(seq.split('/')[:root_folder_id])
    # pdb.set_trace()
    dataset_path = os.path.join(root_path, args.prefix)


    batch_name, place, seq_name = seq.split('/')[root_folder_id+1:]

    merged = {}
    for key in ['image_path', 'mask_path', 'npz_name', 'meta', 'keypoints2d', 'keypoints3d']:
        merged[key] = []
    merged['smpl'] = {}
    for key in ['transl', 'global_orient', 'betas', 'body_pose']:
        merged['smpl'][key] = []
    merged['smplx'] = {}
    for key in ['transl', 'global_orient', 'betas', 'body_pose', 
                'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
        merged['smplx'][key] = []

    try:
        imgname = _get_imgname(seq)
        exrname = _get_exrname(seq)
        valid_frame_number = len(imgname)
        # for p in tqdm(glob.glob(v + '/smpl_with_joints/*.npz'), desc='person'):
        ps = [p for p in os.listdir(os.path.join(seq, 'smpl_withJoints_inCamSpace')) if p.endswith('.npz')]

        for p in sorted(ps):
            npfile_tmp = np.load(os.path.join(seq, 'smpl_withJoints_inCamSpace', p), allow_pickle=True)
            merged['image_path'] += imgname
            merged['mask_path'] += exrname
            merged['npz_name'] += _get_npzname(p, valid_frame_number)

            for _ in range(valid_frame_number):
                merged['meta'].append(npfile_tmp['meta'])

            for key in ['betas', 'global_orient', 'transl', 'body_pose']:
                if key == 'betas' and len(npfile_tmp['smpl'].item()['betas']) == 1:
                    betas = np.repeat(npfile_tmp['smpl'].item()[key], valid_frame_number, axis=0)
                    merged['smpl']['betas'].append(betas)
                else:
                    if len(npfile_tmp['smpl'].item()[key]) == valid_frame_number:
                        merged['smpl'][key].append(npfile_tmp['smpl'].item()[key])
                    else:
                        merged['smpl'][key].append(npfile_tmp['smpl'].item()[key][1:valid_frame_number+1])

            npfile_tmp = np.load(os.path.join(seq, 'smplx_withJoints_inCamSpace', p), allow_pickle=True)
            merged['keypoints2d'].append(npfile_tmp['keypoints2d'][1:valid_frame_number+1])
            merged['keypoints3d'].append(npfile_tmp['keypoints3d'][1:valid_frame_number+1])

            for key in ['betas', 'global_orient', 'transl', 'body_pose', \
                        'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
                if key == 'betas' and len(npfile_tmp['smplx'].item()['betas']) == 1:
                    betas = np.repeat(npfile_tmp['smplx'].item()[key], valid_frame_number, axis=0)
                    merged['smplx']['betas'].append(betas)
                else:
                    if len(npfile_tmp['smplx'].item()[key]) == valid_frame_number:
                        merged['smplx'][key].append(npfile_tmp['smplx'].item()[key])
                    else:
                        merged['smplx'][key].append(npfile_tmp['smplx'].item()[key][1:valid_frame_number+1])

        for k in merged['smpl'].keys():
            merged['smpl'][k] = np.vstack(merged['smpl'][k])
        for k in merged['smplx'].keys():
            merged['smplx'][k] = np.vstack(merged['smplx'][k])
        for k in ['left_hand_pose', 'right_hand_pose']:
            merged['smplx'][k] = merged['smplx'][k].reshape(-1, 15, 3)
        merged['smplx']['body_pose'] = merged['smplx']['body_pose'].reshape(-1, 21, 3)

        merged['keypoints3d'] = np.vstack(merged['keypoints3d'])
        merged['keypoints2d'] = np.vstack(merged['keypoints2d'])

        merged['conf'] = np.vstack(_get_mask_conf(dataset_path, merged)).reshape(-1, 144, 1)

        os.makedirs(os.path.join(args.output_path, batch_name, place), exist_ok=True)
        outpath = os.path.join(args.output_path, batch_name, place, f'{seq_name}.npz')
        np.savez(outpath, **merged)
            
        # pdb.set_trace()
            
    # except Exception as e:
    except Exception as e:
        # failed.append(seq)
        # with open('log_synbody.json', 'w') as f:
        #     json.dump(failed, f)
        print(f'{batch_name}, {place}, {seq_name}', 'failed because of', e)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert datasets')

    parser.add_argument(
        '--seq',
        type=str,
        required=True,
        help='absolute path to the sequence')

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

