# CLIFF

## Introduction

We provide the config files for CLIFF: [CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation](https://arxiv.org/pdf/2208.00571.pdf).

```BibTeX

@Inproceedings{li2022cliff,
  author    = {Li, Zhihao and
               Liu, Jianzhuang and
               Zhang, Zhensong and
               Xu, Songcen and
               Yan, Youliang},
  title     = {CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation},
  booktitle = {ECCV},
  year      = {2022}
}

```

## Notes

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [pascal_occluders.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/pascal_occluders.npy?versionId=CAEQOhiBgMCH2fqigxgiIDY0YzRiNThkMjU1MzRjZTliMTBhZmFmYWY0MTViMTIx)
- [resnet50_a1h2_176-001a1197.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h2_176-001a1197.pth)
- [resnet50_a1h2_176-001a1197.pth(alternative download link)](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/cliff/resnet50_a1h2_176-001a1197.pth)

Download the above resources and arrange them in the following file structure:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── checkpoints
    │   ├── resnet50_a1h2_176-001a1197.pth
    ├── body_models
    │   ├── J_regressor_extra.npy
    │   ├── J_regressor_h36m.npy
    │   ├── smpl_mean_params.npz
    │   └── smpl
    │       ├── SMPL_FEMALE.pkl
    │       ├── SMPL_MALE.pkl
    │       └── SMPL_NEUTRAL.pkl
    ├── preprocessed_datasets
    │   ├── cliff_coco_train.npz
    │   ├── cliff_mpii_train.npz
    │   ├── h36m_mosh_train.npz
    │   ├── muco3dhp_train.npz
    │   ├── mpi_inf_3dhp_train.npz
    │   └── pw3d_test.npz
    ├── occluders
    │   ├── pascal_occluders.npy
    └── datasets
        ├── coco
        ├── h36m
        ├── muco
        ├── mpi_inf_3dhp
        ├── mpii
        └── pw3d
```

## Training
Stage 1: First use [resnet50_pw3d_cache.py](resnet50_pw3d_cache.py) to train.

Stage 2: After around 150 epoches, switch to [resume.py](resume.py) by using "--resume-from" optional argument.

## Results and Models

We evaluate HMR on 3DPW. Values are MPJPE/PA-MPJPE.

|                          Config                           |     3DPW      | Download |
|:---------------------------------------------------------:|:-------------:|:------:|
| Stage 1: [resnet50_pw3d_cache.py](resnet50_pw3d_cache.py) | 48.65 / 76.49 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/cliff/resnet50_cliff-8328e2e2_20230327.pth) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/cliff/20220909_142945.log)
| Stage 2: [resnet50_pw3d_cache.py](resnet50_pw3d_cache.py) | 47.38 / 75.08 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/cliff/resnet50_cliff_new-1e639f1d_20230327.pth) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/cliff/20230222_092227.log)
