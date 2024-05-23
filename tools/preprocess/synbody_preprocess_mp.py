import argparse
import glob
import json
import multiprocessing as mp
import os
import pdb
import time
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.data.data_structures.human_data import HumanData


def process_npz_one(seq):
    # cmd = f'srun -p Zoetrope --gres=gpu:0 --cpus-per-task=4 -x SH-IDC1-10-198-8-[51,56,68,72,78,100,116,123] -N 1 ' \
    #     f'python tools/preprocess/synbody_preprocess.py --seq {seq} ' \
    #     f'--output_path {output_path} ' \
    #     f'--prefix {prefix}'

    cmd = f'python tools/preprocess/synbody_preprocess.py --seq {seq} ' \
        f'--output_path {output_path} ' \
        f'--prefix {prefix}'

    os.system(cmd)


def process_npz_multiprocessing(args):

    # root_path is where the npz files stored. Should ends with 'synbody' or 'Synbody'
    # if not os.path.basename(root_path).endswith('synbody'):
    #     root_path = os.path.join(root_path, 'synbody')
    # ple = [p for p in ple if '.' not in p]

    SUPPORTED_BATCH = ['Synbody_v1_0', 'Synbody_v1_1']
    assert args.prefix in SUPPORTED_BATCH, f'prefix {args.prefix} not supported'

    dataset_path = os.path.join(args.root_path, args.prefix)
    batch_paths = [
        os.path.join(dataset_path, p) for p in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, p))]

    seqs_targeted = glob.glob(
        os.path.join(args.root_path, args.prefix, '*', '*', '*_*'))
    seqs_targeted = sorted(seqs_targeted)
    # pdb.set_trace()
    print(
        f'There are {len(batch_paths)} batches and {len(seqs_targeted)} sequences'
    )

    os.makedirs(args.output_path, exist_ok=True)

    with mp.Pool(args.num_proc) as p:
        r = list(
            tqdm(
                p.imap(process_npz_one, seqs_targeted),
                total=len(seqs_targeted),
                desc='sequences'))


def parse_args():
    parser = argparse.ArgumentParser(description='Convert datasets')

    parser.add_argument(
        '--root_path',
        type=str,
        required=False,
        default='/mnt/lustre/weichen1/datasets/synbody',
        help='the root path of original dataset')

    parser.add_argument(
        '--output_path',
        type=str,
        required=False,
        default='/mnt/lustre/weichen1/datasets/synbody/preprocess',
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
        default=4,
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
