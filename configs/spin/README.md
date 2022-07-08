# SPIN

## Introduction

We provide the config files for SPIN: [Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop](https://arxiv.org/pdf/1909.12828.pdf).

```BibTeX
@inproceedings{SPIN,
  author    = {Nikos Kolotouros and
               Georgios Pavlakos and
               Michael J. Black and
               Kostas Daniilidis},
  title     = {Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop},
  booktitle = {ICCV},
  year      = {2019}
}
```

## Notes

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)
- [gmm_08.pkl](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gmm_08.pkl?versionId=CAEQHhiBgIDP6c3V6xciIGU4ZWFlYzlhNDJmODRmOGViYTMzOGRmODg2YjQ4NTg1)
- Static fits required for the training:
    - [coco_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/coco_fits.npy?versionId=CAEQHhiBgMCr4ZvV6xciIGY1OTZjM2NlZWI3ZDRjMzI5ODE0MWQxYjM2M2Y4NTVk)
    - [h36m_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/h36m_fits.npy?versionId=CAEQHhiBgIC54ZvV6xciIDc2YjExNmM0NjBiMDQwMmU5NjJmODljNjgxYWE1MGQx)
    - [lspet_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/lspet_fits.npy?versionId=CAEQHhiBgIDy4ZvV6xciIDkyMjQ3OGM2YWU5YTRlNTI4MmYxM2I5Njg1Yzc3OWYw)
    - [lsp_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/lsp_fits.npy?versionId=CAEQHhiBgIDS4ZvV6xciIGFmMzdhMWJjZWQ1MjRkODBiZDY3NGU0MTc1Yzg0Nzlh)
    - [mpii_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/mpii_fits.npy?versionId=CAEQHhiBgMCm4ZvV6xciIGM1OTIzZDlkNjVhODQ2MDY5ODkyZWE4ZDEzZGJlNTdi)
    - [mpi_inf_3dhp_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/mpi_inf_3dhp_fits.npy?versionId=CAEQHhiBgMDf4ZvV6xciIDQyYjRmMzdhODdmNDQ1YTBhYzY4MTk1OTAxNzc4MmVj)

As for pretrained model (spin_pretrain.pth), checkpoints of HMR can be directly used. You can download it from [here](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/hmr/resnet50_hmr_pw3d-04f40f58_20211201.pth?versionId=CAEQHhiBgMD6zJfR6xciIDE0ODQ3OGM2OWJjMTRlNmQ5Y2ZjMWZhMzRkOTFiZDFm) and rename it as "spin_pretrain.pth" (or you can change the path of pretrained model in the config).

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
    │   └── spin_pretrain.pth
    ├── preprocessed_datasets
    │   ├── spin_coco_2014_train.npz
    │   ├── h36m_mosh_train.npz
    │   ├── spin_lspet_train.npz
    │   ├── spin_lsp_train.npz
    │   ├── spin_mpi_inf_3dhp_train.npz
    │   ├── spin_mpii_train.npz
    │   └── spin_pw3d_test.npz
    └── static_fits
    │   ├── coco_fits.npy
    │   ├── h36m_fits.npy
    │   ├── lspet_fits.npy
    │   ├── lsp_fits.npy
    │   ├── mpii_fits.npy
    │   └── mpi_inf_3dhp_fits.npy
    └── datasets
        ├── coco
        ├── h36m
        ├── lspet
        ├── lsp
        ├── mpi_inf_3dhp
        ├── mpii
        └── pw3d

```


## Results and Models

We evaluate SPIN on 3DPW. Values are MPJPE/PA-MPJPE.

| Config | 3DPW    | Download |
|:------:|:-------:|:------:|
| [resnet50_spin_pw3d_cache.py](resnet50_spin_pw3d_cache.py) | 94.11 / 57.54 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/resnet50_spin_pw3d-e1d70119_20220708.pth?versionId=CAEQRBiBgMC9xJDujhgiIDQwMGY5Nzc0MDY3YzQzM2U4MmJiMWJiZmRlZWMzOWZh) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/resnet50_spin_pw3d.log?versionId=CAEQRBiBgIDYwZDujhgiIDMxYjFjYTQ2NTI3MzQzNTdiYTU0NjM4N2I2ODQzY2E1) |
