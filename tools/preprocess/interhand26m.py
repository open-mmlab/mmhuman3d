import argparse
import glob
import json
import bigjson
import os

from tqdm import tqdm

import pdb
# from memory_profiler import profile

# @profile
def rewrite_anno_json(args):

    anno_bp = args.dataset_path

    anno_ps = glob.glob(os.path.join(anno_bp, 'anno*30*', '*', '*data.json'))

    for annop in anno_ps:

        with open(annop, 'r') as f:
            anno_data = bigjson.load(f)

        # pdb.set_trace()

            image_data = {}
            for idx in range(len(anno_data['images'])):

                info_slice = anno_data['images'][idx]
                anno_slice = anno_data['annotations'][idx]

                assert anno_slice['image_id'] == info_slice['id']

                imgp = info_slice['file_name']

                del info_slice['file_name']
                del anno_slice['id']

                image_data[imgp] = info_slice | anno_slice

        # save json
        annop_new = annop.replace('.json', '_reformat.json')
        break

        # pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interhand26m preprocess - write dataset format')
    parser.add_argument(
        '--dataset_path', type=str, required=True, help='path to the dataset')

    args = parser.parse_args()
    rewrite_anno_json(args)
