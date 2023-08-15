import argparse
import glob
import os
import pdb

from tqdm import tqdm

from mmhuman3d.data.data_structures.human_data import HumanData


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

    # prepare humandata paths
    src_ps = glob.glob(src)

    # init humandata
    # human_data_s = HumanData()

    # sample
    slices = []
    for src_idx, srcp in enumerate(src_ps):
        human_data = HumanData.fromfile(srcp)

        for idx, k in enumerate(
                tqdm(human_data[key]),
                desc=f'Sampling from npz files {src_idx + 1} / {len(src_ps)}'):
            if k in param:
                selected_slice = human_data.get_slice(idx, idx + 1)
                slices.append(selected_slice)

    # concat slices to humandata
    slice_sample = selected_slice[0]
    # keys = list(slice_sample.keys())

    human_data_s = slice_sample
    for s in tqdm(slices[1:], desc='Concatenating slices'):
        human_data_s = human_data_s.concatenate(human_data_s, s)

    # save humandata
    human_data_s.dump(dst)


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

    # prepare to sample
    sample_humandata_from_key_param(
        src=os.path.join(base_dir, args.humandata_src),
        dst=os.path.join(output_dir, fn).replace('.npz',
                                                 f'_{criteria}_sampled.npz'),
        key=criteria,
        param=c_param)

    pdb.set_trace()


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
