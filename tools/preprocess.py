import argparse
import os

from mmhuman3d.data.preprocessors.coco_pre import coco_extract
from mmhuman3d.data.preprocessors.h36m_pre import h36m_extract
from mmhuman3d.data.preprocessors.mpi_inf_3dhp_pre import mpi_inf_3dhp_extract
from mmhuman3d.data.preprocessors.mpii_pre import mpii_extract
from mmhuman3d.data.preprocessors.pw3d_pre import pw3d_extract


def parse_args():
    parser = argparse.ArgumentParser(description='datasets for preprocessing')

    parser.add_argument(
        '--root_path',
        type=str,
        required=True,
        help='the root path of original data')

    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='the path to store the preprocessed npz files')

    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        required=True,
        default=[],
        help='please offer the dataset names you want to preprocess')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    root_path = args.root_path
    output_path = args.output_path

    if args.datasets == 'all':
        args.datasets = ['coco', '3dpw', 'mpii', 'h36m']

    if 'coco' in args.datasets:
        print('******')
        print('Preprocessing coco ...')
        COCO_ROOT = os.path.join(root_path, 'coco')
        coco_extract(COCO_ROOT, output_path)
        print('Coco preprocess finished!')

    if '3dpw' in args.datasets:
        print('******')
        print('Preprocessing 3DPW ...')
        PW3D_ROOT = os.path.join(root_path, '3DPW')
        pw3d_extract(PW3D_ROOT, output_path, mode='train')
        pw3d_extract(PW3D_ROOT, output_path, mode='test')
        print('3DPW preprocess finished!')

    if 'mpii' in args.datasets:
        print('******')
        print('Preprocessing mpii ...')
        MPII_ROOT = os.path.join(root_path, 'mpii')
        mpii_extract(MPII_ROOT, output_path)
        print('Mpii preprocess finished!')

    if 'h36m' in args.datasets:
        print('******')
        print('Preprocessing h36m ...')
        H36M_ROOT = os.path.join(root_path, 'h36m')
        h36m_extract(H36M_ROOT, output_path, mode='train', protocol=1)
        h36m_extract(H36M_ROOT, output_path, mode='valid', protocol=1)
        h36m_extract(H36M_ROOT, output_path, mode='valid', protocol=2)
        print('H36m preprocess finished!')

    if 'mpi_inf_3dhp' in args.datasets:
        print('******')
        print('Preprocessing mpi_inf_3dhp ...')
        MPI_INF_3DHP_ROOT = os.path.join(root_path, 'mpi_inf_3dhp')
        mpi_inf_3dhp_extract(MPI_INF_3DHP_ROOT, output_path, 'train')
        mpi_inf_3dhp_extract(MPI_INF_3DHP_ROOT, output_path, 'test')
        print('Mpi_inf_3dhp preprocess finished!')


if __name__ == '__main__':
    main()
