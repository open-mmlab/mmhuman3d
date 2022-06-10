# PARE

## Introduction

We provide the config files for PARE: [Part Attention Regressor for 3D Human Body Estimation](https://arxiv.org/abs/2104.08527).

```BibTeX
@inproceedings{Kocabas_PARE_2021,
  title = {{PARE}: Part Attention Regressor for {3D} Human Body Estimation},
  author = {Kocabas, Muhammed and Huang, Chun-Hao P. and Hilliges, Otmar and Black, Michael J.},
  booktitle = {Proc. International Conference on Computer Vision (ICCV)},
  pages = {11127--11137},
  month = oct,
  year = {2021},
  doi = {},
  month_numeric = {10}
}
```

## Notes

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)
- Pascal Occluders for the pretraining:
  - [pascal_occluders.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/pascal_occluders.npy?versionId=CAEQOhiBgMCH2fqigxgiIDY0YzRiNThkMjU1MzRjZTliMTBhZmFmYWY0MTViMTIx)

As for pretrained model (hrnet_w32_conv_pare_coco.pth). You can download it from [here](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/hrnet_w32_conv_pare_coco.pth?versionId=CAEQOhiBgMCxmv_RgxgiIDkxNWJhOWMxNDEyMzQ1OGQ4YTQ3NjgwNjA0MWUzNDE5) and change the path of pretrained model in the config.
You can also pretrain the model using [hrnet_w32_conv_pare_coco.py]([hrnet_w32_conv_pare_coco.py]). Download the hrnet pretrain from [here](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/hrnet_pretrain.pth?versionId=CAEQOhiBgMC26fSigxgiIGViMTFiZmJkZDljMDRhMWY4Mjc5Y2UzNzBmYzU1MGVk
) for pretrain.

Download the above resources and arrange them in the following file structure:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── gmm_08.pkl
    ├── body_models
    │   ├── J_regressor_extra.npy
    │   ├── J_regressor_h36m.npy
    │   ├── smpl_mean_params.npz
    │   └── smpl
    │       ├── SMPL_FEMALE.pkl
    │       ├── SMPL_MALE.pkl
    │       └── SMPL_NEUTRAL.pkl
    ├── pretrained_models
    │   ├── hrnet_pretrain.pth
    │   └── hrnet_w32_conv_pare_coco.pth
    ├── preprocessed_datasets
    │   ├── h36m_mosh_train.npz
    │   ├── h36m_train.npz
    │   ├── mpi_inf_3dhp_train.npz
    │   ├── eft_mpii.npz
    │   ├── eft_lspet.npz
    │   ├── eft_coco_all.npz
    │   ├── pw3d_test.npz
    ├── occluders
    │   ├── pascal_occluders.npy
    └── datasets
        ├── coco
        ├── h36m
        ├── lspet
        ├── mpi_inf_3dhp
        ├── mpii
        └── pw3d

```


## Results and Models

We evaluate PARE on 3DPW. Values are MPJPE/PA-MPJPE.

Trained with MoShed Human3.6M Datasets and Cache:

| Config | 3DPW    | Download |
|:------:|:-------:|:------:|
| [hrnet_w32_conv_pare_mix_cache.py](hrnet_w32_conv_pare_mix_cache.py) | 81.79 / 49.35 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/with_mosh/hrnet_w32_conv_pare_mosh.pth?versionId=CAEQOhiBgIDooeHSgxgiIDkwYzViMTUyNjM1MjQ3ZDNiNzNjMjJlOGFlNjgxYjlh) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/with_mosh/20220427_113717.log?versionId=CAEQOhiBgMClqr3PgxgiIGRjZWU0NzFhMmVkMDQzN2I5ZmY5Y2MxMzJiZDM3MGQ0) |


Trained without MoShed Human3.6M Datasets:
| Config | 3DPW    | Download |
|:------:|:-------:|:------:|
| [hrnet_w32_conv_pare_mix_no_mosh.py](hrnet_w32_conv_pare_mix_no_mosh.py) | 81.81 / 50.78 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/without_mosh/hrnet_w32_conv_pare.pth?versionId=CAEQOhiBgMCi4YbVgxgiIDgzYzFhMWNlNDE2NTQwN2ZiOTQ1ZGJmYTM4OTNmYWY5) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/without_mosh/20220427_113844.log?versionId=CAEQOhiBgMCHwcTPgxgiIGI0NjI0M2JiM2ViMzRhMTFiMWQxZDJmMGI5MmQwMjgw) |
