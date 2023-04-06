import argparse
import os

from mmhuman3d.data.data_converters import build_data_converter

DATASET_CONFIGS = dict(
    agora=dict(
        type='AgoraConverter', modes=['train', 'validation'], fit='smplx'),
    amass=dict(type='AmassConverter', prefix='AMASS_file'),
    coco=dict(type='CocoConverter'),
    coco_wholebody=dict(
        type='CocoWholebodyConverter', modes=['train', 'val'], prefix='coco'),
    crowdpose=dict(
        type='CrowdposeConverter', modes=['train', 'val', 'test', 'trainval']),
    pw3d=dict(type='Pw3dConverter', modes=['train', 'test']),
    mpii=dict(type='MpiiConverter'),
    h36m_p1=dict(
        type='H36mConverter',
        modes=['train', 'valid'],
        protocol=1,
        mosh_dir='data/datasets/h36m_mosh',
        prefix='h36m'),
    h36m_p2=dict(
        type='H36mConverter', modes=['valid'], protocol=2, prefix='h36m'),
    mpi_inf_3dhp=dict(type='MpiInf3dhpConverter', modes=['train', 'test']),
    penn_action=dict(type='PennActionConverter'),
    lsp_original=dict(type='LspConverter', modes=['train'], prefix='lsp'),
    lsp_dataset=dict(type='LspConverter', modes=['test']),
    lsp_extended=dict(type='LspExtendedConverter', prefix='lspet'),
    up3d=dict(
        type='Up3dConverter', modes=['trainval', 'test'], prefix='up-3d'),
    posetrack=dict(type='PosetrackConverter', modes=['train', 'val']),
    instavariety_vibe=dict(type='InstaVibeConverter', prefix='vibe_data'),
    eft=dict(
        type='EftConverter', modes=['coco_all', 'coco_part', 'mpii', 'lspet']),
    coco_hybrik=dict(type='CocoHybrIKConverter', prefix='coco/train_2017'),
    pw3d_hybrik=dict(type='Pw3dHybrIKConverter', prefix='hybrik_data'),
    h36m_hybrik=dict(
        type='H36mHybrIKConverter',
        modes=['train', 'test'],
        prefix='hybrik_data'),
    mpi_inf_3dhp_hybrik=dict(
        type='MpiInf3dhpHybrIKConverter',
        modes=['train', 'test'],
        prefix='hybrik_data'),
    surreal=dict(
        type='SurrealConverter', modes=['train', 'val', 'test'], run=0),
    spin=dict(
        type='SpinConverter',
        modes=['coco_2014', 'lsp', 'mpii', 'mpi_inf_3dhp', 'lspet'],
        prefix='spin_data'),
    vibe=dict(
        type='VibeConverter',
        modes=['pw3d', 'mpi_inf_3dhp'],
        pretrained_ckpt='data/checkpoints/spin.pth',
        prefix='vibe_data'),
    gta_human=dict(type='GTAHumanConverter', prefix='gta_human'),
    humman=dict(
        type='HuMManConverter', modes=['train', 'test'], prefix='humman'),
    cliff=dict(type='CliffConverter', modes=['coco', 'mpii']))


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

    parser.add_argument(
        '--enable_multi_human_data',
        type=bool,
        default=False,
        help='Whether to generate a multi-human data')

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
        data_converter.convert(
            input_path,
            args.output_path,
            enable_multi_human_data=args.enable_multi_human_data)
        print(f'[{dataset}] Converting finished!')


if __name__ == '__main__':
    main()
