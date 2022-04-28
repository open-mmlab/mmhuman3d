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
As for pretrained model (hrnet_w32_conv_pare_coco.pth). You can download it from [here]() and change the path of pretrained model in the config.
You can also pretrain the model using [hrnet_w32_conv_pare_coco.py]([hrnet_w32_conv_pare_coco.py]). Then download the hrnet pretrain from [here](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/hrnet_pretrain.pth?versionId=CAEQOhiBgMC26fSigxgiIGViMTFiZmJkZDljMDRhMWY4Mjc5Y2UzNzBmYzU1MGVk
)

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
    ├── pretrained
    │   ├── hrnet_pretrain.pth
    │   └── pare_pretrain.pth

    ├── preprocessed_datasets
    │   ├── h36m_mosh_train.npz
    │   ├── mpi_inf_3dhp_train.npz
    │   ├── eft_mpii.npz
    │   ├── eft_lspet.npz
    │   ├── eft_coco_all.npz
    |   ├── pw3d_test.npz
    └── occluders
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

| Config | 3DPW    | Download |
|:------:|:-------:|:------:|
| [hrnet_w32_conv_pare_mix.py](hrnet_w32_conv_pare_mix.py) | 81.74 / 48.69 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/resnet50_spin_pw3d-e1857270_20211201.pth?versionId=CAEQHhiBgMDyvYnS6xciIDZhNTg4NmM4OGE4MTQ0ODRhY2JlY2JmZDI4ZWQ0ZmU3) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/20211129_160633.log?versionId=CAEQHhiBgICCvYnS6xciIDIwMmVlNjZiYzFjOTQ1ZjBiMjg3NTJkY2U5YWMwZDJl) |
