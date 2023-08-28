import argparse
import glob
import json
import os
import pdb

from tqdm import tqdm

# from memory_profiler import profile


# @profile
def mscoco_rewrite_anno_json(args):

    anno_bp = args.dataset_path

    anno_bp = os.path.join(anno_bp, 'annotations', '*train*.json')

    anno_ps = glob.glob(anno_bp)
    anno_ps = [x for x in anno_ps if 'SMPLX' not in x]
    anno_ps = [x for x in anno_ps if 'reformat' not in x]
    print(anno_ps)

    for annop in anno_ps:

        with open(annop, 'r') as f:
            anno_data = json.load(f)

        image_data = {}

        image_info_dict = {}
        anno_info_dict = {}

        for idx in tqdm(
                range(len(anno_data['images'])), desc='extracting image info'):

            image_info_slice = anno_data['images'][idx]
            image_info_dict[image_info_slice['id']] = image_info_slice

        for idx in tqdm(
                range(len(anno_data['annotations'])),
                desc='extracting anno info'):

            anno_info_slice = anno_data['annotations'][idx]
            anno_info_dict[idx] = anno_info_slice

        for id in tqdm(anno_info_dict.keys(), desc='merging info'):

            imid = anno_info_dict[id]['image_id']

            if imid not in image_info_dict.keys():
                continue

            info_slice = image_info_dict[imid]
            anno_slice = anno_info_dict[id]

            try:
                imgp = os.path.basename(info_slice['coco_url'])
                del info_slice['id']
                del info_slice['file_name']
            except KeyError:
                continue

            aid = anno_slice['id']

            image_data[aid] = info_slice | anno_slice
            image_data[aid]['image_name'] = imgp

        annop_new = annop.replace('.json', '_reformat.json')
        json.dump(image_data, open(annop_new, 'w'))


def pw3d_rewrite_anno_json(args):

    anno_bp = args.dataset_path

    anno_bp = os.path.join(anno_bp, '*.json')

    anno_ps = glob.glob(anno_bp)
    anno_ps = [x for x in anno_ps if 'SMPLX' not in x]
    anno_ps = [x for x in anno_ps if 'reformat' not in x]
    print(anno_ps)

    for annop in anno_ps:

        with open(annop, 'r') as f:
            anno_data = json.load(f)

        image_data = {}

        image_info_dict = {}
        anno_info_dict = {}

        for idx in tqdm(
                range(len(anno_data['images'])), desc='extracting image info'):

            image_info_slice = anno_data['images'][idx]
            image_info_dict[image_info_slice['id']] = image_info_slice

        for idx in tqdm(
                range(len(anno_data['annotations'])),
                desc='extracting anno info'):

            anno_info_slice = anno_data['annotations'][idx]
            anno_info_dict[idx] = anno_info_slice

        for id in tqdm(anno_info_dict.keys(), desc='merging info'):

            imid = anno_info_dict[id]['image_id']

            if imid not in image_info_dict.keys():
                continue

            info_slice = image_info_dict[imid]
            anno_slice = anno_info_dict[id]

            if 'file_name' not in info_slice.keys():
                file_name = 'image_{:05d}.jpg'.format(info_slice['frame_idx'])
                imgp = os.path.join(info_slice['sequence'], file_name)
            else:
                imgp = os.path.join(info_slice['sequence'],
                                    info_slice['file_name'])
                del info_slice['file_name']
            if 'id' in info_slice.keys():
                del info_slice['id']

            aid = anno_slice['id']

            image_data[aid] = info_slice | anno_slice
            image_data[aid]['image_name'] = imgp
        # pdb.set_trace()
        annop_new = annop.replace('.json', '_reformat.json')
        json.dump(image_data, open(annop_new, 'w'))


def mpii_rewrite_anno_json(args):

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

        for idx in tqdm(
                range(len(anno_data['images'])), desc='extracting image info'):

            image_info_slice = anno_data['images'][idx]
            image_info_dict[image_info_slice['id']] = image_info_slice

        for idx in tqdm(
                range(len(anno_data['annotations'])),
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

        # save json
        annop_new = annop.replace('.json', '_reformat.json')
        json.dump(image_data, open(annop_new, 'w'))
        # break

        # pdb.set_trace()


def main(args):

    dataset_name = os.path.basename(args.dataset_path)

    SUPPORTED_DATASETS = ['mscoco', 'pw3d', 'mpii']
    assert dataset_name in SUPPORTED_DATASETS, \
        f'Only support {SUPPORTED_DATASETS}, got {dataset_name}'

    if dataset_name == 'mscoco':
        mscoco_rewrite_anno_json(args)

    if dataset_name == 'pw3d':
        pw3d_rewrite_anno_json(args)

    if dataset_name == 'mpii':
        mpii_rewrite_anno_json(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Neural annot dataset preprocess - rewrite dataset format')
    parser.add_argument(
        '--dataset_path', type=str, required=True, help='path to the dataset')
    # python tools/preprocess/neural_annot.py --dataset_path /mnt/d/mscoco/annotations
    args = parser.parse_args()
    main(args)
