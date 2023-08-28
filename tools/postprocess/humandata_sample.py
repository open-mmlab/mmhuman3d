import argparse
import glob
import os
import pdb
import json
import numpy as np

from tqdm import tqdm
import time

from mmhuman3d.data.data_structures.human_data import HumanData


def get_slice(human_data, index):
    # pdb.set_trace()
    h_slice = {}

    for key in ['keypoints2d_smplx_mask', 'keypoints3d_smplx_mask', 
                'misc', 'config', 'keypoints2d_smplx_convention', 
                'keypoints3d_smplx_convention']:
        h_slice[key] = human_data[key]

    for key in ['image_path', 'bbox_xywh', 'face_bbox_xywh', 
                'lhand_bbox_xywh', 'rhand_bbox_xywh', 
                'keypoints2d_smplx', 'keypoints3d_smplx']:
        h_slice[key] = human_data[key][index:index + 1]

    for key in ['smplx', 'meta']:
        dict_slice = {}
        for k in human_data[key].keys():
            dict_slice[k] = human_data[key][k][index:index + 1]
        h_slice[key] = dict_slice
    # pdb.set_trace()
    return h_slice


def slice_to_humandata(slices, dst):
    human_data = HumanData()

    for key in ['keypoints2d_smplx_mask', 'keypoints3d_smplx_mask', 
                'misc', 'config', 'keypoints2d_smplx_convention', 
                'keypoints3d_smplx_convention']:
        human_data[key] = slices[0][key]

    for key in ['image_path']:
        image_paths = []
        for s in slices:
            image_paths += [list(imgp)[0] for imgp in list(s[key])]
        human_data[key] = image_paths

    for key in [ 'bbox_xywh', 'face_bbox_xywh', 
                'lhand_bbox_xywh', 'rhand_bbox_xywh', 
                'keypoints2d_smplx', 'keypoints3d_smplx']:
        human_data[key] = np.concatenate([s[key] for s in slices], axis=0)

    # pdb.set_trace()
    for key in ['smplx', 'meta']:
        dict_slice = {}
        for k in slices[0][key].keys():
           dict_slice[k] = np.concatenate([s[key][k] for s in slices], axis=0)
        human_data[key] = dict_slice
    human_data.dump(dst)


def sample_humandata_from_key_param(src, dst, key, param) -> None:
    """This function is used to sample selected HumanData from all HumanData,
    based on given keys and parameters.

    Inputs:
        - src: str, the path of the source HumanData (glob.glob syntax)
        - dst: str, output path
        - key: str, the key of the HumanData to be sampled according to
        - param: str, the parameter of sample criteria

    Returns:
        - None
    """
    # param = param[:100]
    # prepare humandata paths
    src_ps = glob.glob(src)
    print(f'Looking for {len(param)} slices in {len(src_ps)} HumanData files.')

    # init humandata
    # human_data_s = HumanData()

    # sample

    # for src_idx, srcp in enumerate(src_ps):
    #     human_data = HumanData.fromfile(srcp)

    #     unmatched_slice = len(param)
    #     total_src = len(human_data[key])

    #     # divide into 100 sub slices
    #     sub_slices = 100
    #     slice = int(int(total_src) / sub_slices) + 1
    #     # sub_slices = 5
    #     for i in tqdm(range(sub_slices), leave=False, position=0,
    #                   desc=f'Sampling from npz files {src_idx + 1} / {len(src_ps)}, In subslices'):
            
    #         human_data_slice = human_data.get_slice(i * slice, (i + 1) * slice)
    #         print(end="\r")
    #         slices = []
    #         for idx, k in enumerate(
    #                 tqdm(human_data_slice[key], leave=False, position=1, total=slice)):
    #             # pdb.set_trace()
    #             if k.replace('20230727/', '') in param:
    #                 # selected_slice = human_data.get_slice(idx, idx + 1)
    #                 selected_slice = get_slice(human_data_slice, idx)
    #                 slices.append(selected_slice)
    #                 # pdb.set_trace()
    #                 param.remove(k.replace('20230727/', ''))
    #                 # print('found')
    #         dst_t = dst.replace('.npz', f'_npz{src_idx}_subslice{i}.npz')
    #         slice_to_humandata(slices, dst_t)
    #         del slices
    #         del human_data_slice
    #     del human_data

    #     print(f'Found {unmatched_slice - len(param)} matching slices in {srcp}.')
    # print(f'Searching slice finished...collecting to HumanData...')

    # with open(dst.replace('.npz', '.json'), 'a') as f:
    #     json.dump(slices, f, indent=1)
    # concat slices to humandata
    
    dst_ps = glob.glob(os.path.dirname(dst) + '/*.npz')

    slices = []
    print('Collecting slices to HumanData...')
    for dstp in  tqdm(dst_ps):
        human_data_s = HumanData.fromfile(dstp)
        slices.append(human_data_s)

    slice_to_humandata(slices, dst)




def main(args):
    """Sample a humandata from humandata based on humandata Currently support
    sampling based on image_path."""
    base_dir = args.humandata_dir
    output_dir = args.output_dir
    fn = args.humandata_sample_base

    # load base humandata
    base_ps = glob.glob(os.path.join(base_dir, fn))
    criteria = args.key
    c_param = []
    for bp in base_ps:
        human_data = HumanData.fromfile(bp)

        c_param += human_data[criteria]
    # pdb.set_trace()
    # prepare to sample
    sample_humandata_from_key_param(
        src=os.path.join(base_dir, args.humandata_src),
        dst=os.path.join(output_dir, 'tmp', fn).replace('.npz', f'_s.npz'),
        #.replace('.npz', f'_{criteria}_sampled.npz'),
        key=criteria, param=sorted(c_param))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--humandata_dir',
        type=str,
        default='/mnt/c/Users/12595/Desktop/synbody_liushuai')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/c/Users/12595/Desktop/synbody_liushuai')

    parser.add_argument(
        '--humandata_src', type=str, default='synbody_v1*38400*.npz')
    parser.add_argument(
        '--humandata_sample_base', type=str, default='synbody_v1_1_10w.npz')

    parser.add_argument('--key', type=str, default='image_path')
    args = parser.parse_args()

    main(args)
