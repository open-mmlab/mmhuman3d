import argparse
import glob
import json
import os

from tqdm import tqdm

import pdb
# from memory_profiler import profile


# @profile
def rewrite_anno_json(args):

    anno_bp = args.dataset_path
    # for interhand2.6m
    # anno_bp = os.path.join(anno_bp, 'anno*5*', '*', '*test*data.json')
    # for blurhand
    anno_bp = os.path.join(anno_bp, 'annotations', 'train.json')

    anno_ps = glob.glob(anno_bp)
    anno_ps = [x for x in anno_ps if 'SMPLX' not in x]
    print(anno_ps)

    for annop in anno_ps:

        with open(annop, 'r') as f:
            anno_data = json.load(f)

        image_data = {}


        image_info_dict = {}
        anno_info_dict = {}

        for idx in tqdm(range(len(anno_data['images'])),
                        desc='extracting image info'):

            image_info_slice = anno_data['images'][idx]
            image_info_dict[image_info_slice['id']] = image_info_slice
            pdb.set_trace()

        for idx in tqdm(range(len(anno_data['annotations'])),
                        desc='extracting anno info'):
                
            anno_info_slice = anno_data['annotations'][idx]
            anno_info_dict[anno_info_slice['image_id']] = anno_info_slice


        for id in tqdm(image_info_dict.keys(), desc='merging info'):

            if id not in anno_info_dict.keys():
                continue

            info_slice = image_info_dict[id]
            anno_slice = anno_info_dict[id]

            imgp = os.path.basename(info_slice['file_name'])

            del info_slice['file_name']
            del info_slice['id']

            image_data[imgp] = info_slice | anno_slice

            pdb.set_trace()
        # save json
        annop_new = annop.replace('.json', '_reformat.json')
        json.dump(image_data, open(annop_new, 'w'))
        # break

        # pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interhand26m preprocess - write dataset format')
    parser.add_argument(
        '--dataset_path', type=str, required=True, help='path to the dataset')
    # python tools/preprocess/mpii.py --dataset_path /mnt/e/mpii
    args = parser.parse_args()
    rewrite_anno_json(args)
