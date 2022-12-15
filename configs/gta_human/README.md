# GTA-Human

## Introduction

We provide the config files for GTA-Human: Playing for 3D Human Recovery.
[[Homepage]](https://caizhongang.github.io/projects/GTA-Human/)
[[Preprint]](https://arxiv.org/abs/2110.07588)

Notes:
- the pretrained models for HMR, SPIN and PARE baselines are currently available.
- more baselines (HMR+ and VIBE) are coming soon.

```BibTeX
@article{GTAHuman,
  author    = {Cai, Zhongang and
               Zhang, Mingyuan and
               Ren, Jiawei and
               Wei, Chen and
               Ren, Daxuan and
               Lin, Zhengyu and
               Zhao, Haiyu and
               Yang, Lei and
               Liu, Ziwei},
  title     = {Playing for 3D human recovery},
  journal   = {arXiv preprint arXiv:2110.07588},
  year      = {2021}
}
```

## Downloads and Data Preparation

| Google Drive | OneDrive (CN) |
|:------:|:-------:|
|[Download](https://drive.google.com/drive/folders/1-vaWuFkYRNyFj1ONaCrYr009KNitHAqA?usp=sharing)|[Download](https://pjlab-my.sharepoint.cn/:f:/g/personal/openmmlab_pjlab_org_cn/EjT3W_PHhApGvDB0geyC_g0BoBPK0tZfLVATnecU_bJl1A?e=rm3tSe)|

The downloaded compressed .zip files consist of:
- 31 image files (total ~280 GB)
- one annotation file (~4 GB, optional)

Download the data files and decompress them under data/datasets/gta/ :

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── preprocessed_datasets
    │   └── gta_human_4x.npz  
    └── datasets
        └── gta
            ├── annotations (optional)
            └── images
```

MMHuman3D uses the standardized [`HumanData`](../../docs/human_data.md) to store annotations.  
Hence, `gta_human_4x.npz` is used as the training, it may be obtained in two ways:
1. (Recommended) Directly download from this [link](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/gta_human_4x.npz?versionId=CAEQRBiBgIDzq.n4jhgiIDJkZDU4OWU5MWEzYzQ5ZjA4ZjJiN2IwNjUzN2FmMTJl)
   as we have already preprocessed it for you.
2. Decompress the annotation file and run data_converter (use `gta_human` for dataset name).
   See [preprocess_dataset.md](../../docs/preprocess_dataset.md) for details.

Please refer to [getting_started.md](../../docs/getting_started.md) for training and evaluation on MMHuman3D.

## Notes

For different base models, you can find detailed data preparation steps in each subfolder.

## Results and Models

We evaluate HMR, SPIN and PARE on 3DPW. Values are MPJPE/PA-MPJPE.

| Config | 3DPW    | Download |
|:------:|:-------:|:------:|
| [resnet50_hmr_gta_bt.py](hmr/resnet50_hmr_gta_bt.py) | 98.72 / 60.49 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_hmr_gta_bt-aa643b97_20220416.pth?versionId=CAEQLxiBgIDa4qHFgRgiIGUwNWJjZGFjMDE0OTRjYTg5MjI4MjcyZjI2YTVhMjli) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_hmr_gta_bt.log?versionId=CAEQLxiBgMCN4qHFgRgiIGZkNjJhMWY0YjFhODQxMGY5NTdmNjBhYTUwZDI3MmJj) |
| [resnet50_hmr_gta_ft.py](hmr/resnet50_hmr_gta_ft.py) | 91.42 / 55.71 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_hmr_gta_ft-f444e49c_20220416.pth?versionId=CAEQLxiBgMD04aHFgRgiIDg0YTExY2IzNWFmMjQ3MTc5NDFjY2MyNWU4MmM5Mzcz) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_hmr_gta_ft.log?versionId=CAEQLxiBgID936HFgRgiIDAwMDM5NDlkM2MyNzQxYTE4ZTgzZDc3ZGE4NTJlZTVh) |
| [resnet50_spin_gta_ft.py](spin/resnet50_spin_gta_ft.py) | 83.20 / 51.98 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_spin_gta_ft/resnet50_spin_gta_ft-2538df81_20220708.pth?versionId=CAEQRBiBgICJxdjujhgiIGQwMTcwOGI5YzdlMTQ1ZjVhYzRhNWZkOTVhY2U3NjFm) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_spin_gta_ft/resnet50_spin_gta_ft.log?versionId=CAEQRBiBgMCHrdfujhgiIGRhZDA4NjY0NDBmNDRkMGRhMWRmODZlMzM1YmRiNzRj) |
| [hrnet_w32_conv_pare_gta_ft.py](pare/hrnet_w32_conv_pare_gta_ft.py) | 77.52 / 46.84 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/hrnet_w32_conv_pare_gta_ft/hrnet_w32_conv_pare_gta_ft-838829bc_20220708.pth?versionId=CAEQRBiBgMDRxNjujhgiIGY3ZmUzMjUzZjJhNjQ2MTg5ODNjMWFlNTJmMGJhMmFh) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/hrnet_w32_conv_pare_gta_ft/hrnet_w32_conv_pare_gta_ft.log?versionId=CAEQRBiBgICjxdfujhgiIGZiZDFmMmI1YWI0MzQyZjM4MmQ2MjZiYzY5OGQ5ODk1) |
