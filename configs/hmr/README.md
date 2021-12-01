# HMR

## Introduction

We provide the config files for HMR: [End-to-End Recovery of Human Shape and Pose](https://arxiv.org/pdf/1712.06584.pdf).

```BibTeX
@inproceedings{HMR,
  author    = {Angjoo Kanazawa and
               Michael J. Black and
               David W. Jacobs and
               Jitendra Malik},
  title     = {End-to-End Recovery of Human Shape and Pose},
  booktitle = {CVPR},
  year      = {2018}
}
```

## Notes

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)

Download the above resources and arrange them in the following file structure:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── body_models
    │   ├── J_regressor_extra.npy
    │   ├── J_regressor_h36m.npy
    │   ├── smpl_mean_params.npz
    │   └── smpl
    │       ├── SMPL_FEMALE.pkl
    │       ├── SMPL_MALE.pkl
    │       └── SMPL_NEUTRAL.pkl
    ├── preprocessed_datasets
    │   ├── cmu_mosh.npz
    │   ├── coco_2014_train.npz
    │   ├── h36m_mosh_train.npz
    │   ├── lspet_train.npz
    │   ├── lsp_train.npz
    │   ├── mpi_inf_3dhp_train.npz
    │   ├── mpii_train.npz
    │   └── pw3d_test.npz
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

We evaluate HMR on 3DPW. Values are MPJPE/PA-MPJPE.

| Config | 3DPW    | Download |
|:------:|:-------:|:------:|
| [resnet50_hmr_pw3d.py](resnet50_hmr_pw3d.py) | 112.34 / 67.53 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/hmr/resnet50_hmr_pw3d-04f40f58_20211201.pth?versionId=CAEQHhiBgMD6zJfR6xciIDE0ODQ3OGM2OWJjMTRlNmQ5Y2ZjMWZhMzRkOTFiZDFm) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/hmr/20211128_053633.log?versionId=CAEQHhiBgMDbzZfR6xciIGZkZjM2NWEwN2ExYzQ1NGViNzg2ODA0YTAxMmU4M2Vi) |
