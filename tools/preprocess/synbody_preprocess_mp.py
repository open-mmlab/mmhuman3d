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
import multiprocessing as mp


def process_npz_one(seq):
    cmd = f'srun -p Zoetrope --gres=gpu:0 --cpus-per-task=4 -x SH-IDC1-10-198-8-[51,56,68,72,78,100,116,123] -N 1 ' \
        f'python tools/synbody_preprocess.py --seq {seq} ' \
        f'--output_path {output_path} ' \
        f'--prefix {prefix}'

    os.system(cmd)



def process_npz_multiprocessing(args):

    # root_path is where the npz files stored. Should ends with 'synbody' or 'Synbody'
    # if not os.path.basename(root_path).endswith('synbody'):
    #     root_path = os.path.join(root_path, 'synbody')
    # ple = [p for p in ple if '.' not in p]
    dataset_path = os.path.join(args.root_path, args.prefix)
    batch_paths = [os.path.join(args.root_path, p) for p in os.listdir(dataset_path)]

    seqs_targeted = glob.glob(os.path.join(args.root_path, args.prefix, '*', '*', 'LS*'))
    print(f'There are {len(batch_paths)} batches and {len(seqs_targeted)} sequences')

    # print(ple)
    os.makedirs(args.output_path, exist_ok=True)
    # failed = []

    with mp.Pool(args.num_proc) as p:
        r = list(tqdm(p.imap(process_npz_one, seqs_targeted), total=len(seqs_targeted), desc='sequences'))


def parse_args():
    parser = argparse.ArgumentParser(description='Convert datasets')

    parser.add_argument(
        '--root_path',
        type=str,
        required=True,
        help='the root path of original dataset')

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
    
    parser.add_argument(
        '--num_proc',
        required=False,
        type=int,
        default=1,
        help='num of processes')

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

    output_path = args.output_path
    prefix = args.prefix
    process_npz_multiprocessing(args)
