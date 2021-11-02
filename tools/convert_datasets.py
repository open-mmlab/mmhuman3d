import argparse
import os

from mmhuman3d.data.data_converters import build_data_converter

DATASET_CONFIGS = dict(
    agora=dict(
        type='AgoraConverter', modes=['train', 'validation'], fit='smplx'),
    amass=dict(type='AmassConverter', prefix='AMASS_file'),
    coco=dict(type='CocoConverter'),
    coco_wholebody=dict(type='CocoWholebodyConverter', modes=['train', 'val']),
    crowdpose=dict(
        type='CrowdposeConverter',
        modes=['train', 'val', 'test', 'trainval'],
        prefix='Crowdpose'),
    pw3d=dict(type='Pw3dConverter', modes=['train', 'test'], prefix='3DPW'),
    mpii=dict(type='MpiiConverter'),
    h36m_p1=dict(
        type='H36mConverter',
        modes=['train', 'valid'],
        protocol=1,
        prefix='h36m'),
    h36m_p2=dict(
        type='H36mConverter', modes=['valid'], protocol=2, prefix='h36m'),
    mpi_inf_3dhp=dict(type='MpiInf3dhpConverter', modes=['train', 'test']),
    penn_action=dict(type='PennActionConverter', prefix='Penn_Action'),
    lsp_original=dict(
        type='LspConverter', modes=['train'], prefix='lsp_dataset_original'),
    lsp_dataset=dict(type='LspConverter', modes=['test']),
    lsp_extended=dict(type='LspExtendedConverter', prefix='hr-lspet'),
    up3d=dict(
        type='Up3dConverter', modes=['trainval', 'test'], prefix='up-3d'),
    posetrack=dict(
        type='PosetrackConverter',
        modes=['train', 'val'],
        prefix='PoseTrack/data'),
    eft=dict(
        type='EftConverter', modes=['coco_all', 'coco_part', 'mpii', 'lspet']))


def parse_args():
    parser = argparse.ArgumentParser(description='Convert datasets')

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
        help=f'Supported datasets: {list(DATASET_CONFIGS.keys())}')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    datasets = (
        DATASET_CONFIGS.keys() if args.datasets == ['all'] else args.datasets)

    for dataset in datasets:
        print(f'[{dataset}] Converting ...')
        cfg = DATASET_CONFIGS[dataset]
        prefix = cfg.pop('prefix', dataset)
        input_path = os.path.join(args.root_path, prefix)
        data_converter = build_data_converter(cfg)
        data_converter.convert(input_path, args.output_path)
        print(f'[{dataset}] Converting finished!')


if __name__ == '__main__':
    main()
