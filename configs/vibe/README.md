# VIBE

## Introduction

We provide the config files for VIBE: [VIBE: Video Inference for Human Body Pose and Shape Estimation](https://arxiv.org/pdf/1912.05656.pdf).

```BibTeX
@inproceedings{VIBE,
  author    = {Muhammed Kocabas and
               Nikos Athanasiou and
               Michael J. Black},
  title     = {{VIBE}: Video Inference for Human Body Pose and Shape Estimation},
  booktitle = {CVPR},
  year      = {2020}
}
```

## Notes

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)
- The pretrained frame feature extractor [spin.pth](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/vibe/spin.pth?versionId=CAEQKhiBgMD9hdub9xciIDIxMGI4NmMxMGIyNzQxOGM5YzYxZjMyYTVmMjllOTAy)

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
    ├── pretrained
    │   └── spin.pth
    ├── preprocessed_datasets
    │   ├── vibe_mpi_inf_3dhp_train.npz
    │   └── vibe_insta_variety.npz
    └── datasets
        └── mpi_inf_3dhp
```

## Results and Models

We evaluate VIBE on 3DPW. Values are MPJPE/PA-MPJPE.

| Config | 3DPW    | Download |
|:------:|:-------:|:------:|
| [resnet50_vibe_pw3d.py]() | 94.89 / 57.08 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/vibe/resnet50_vibe_pw3d-2e05a122_20211201.pth?versionId=CAEQHhiBgMCNvonS6xciIGEyOGM1M2M0ZTdiMDQ4NTc4NDI1MjBmYzgyMjUwMWI2) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/vibe/20211201_112201.log?versionId=CAEQHhiBgMDdvInS6xciIDI0Yzg1NzVhZTNjZDQ1Nzg4MDAyZDE5NTllYTM5ZmU2) |
