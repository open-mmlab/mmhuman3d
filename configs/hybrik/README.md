# HybrIK

## Introduction

We provide the config files for HybrIK: [HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation](https://arxiv.org/pdf/2011.14672.pdf).

```BibTeX
@inproceedings{HybrIK,
  author    = {Jiefeng Li and
               Chao Xu and
               Zhicun Chen and
               Siyuan Bian and
               Lixin Yang and
               Cewu Lu},
  title     = {{HybrIK}: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation},
  booktitle = {CVPR},
  year      = {2021}
}
```

## Notes

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_beta.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/hybrik/h36m_mean_beta.npy?versionId=CAEQHhiBgMDnt_DV6xciIGM5MzM0MGI1NzBmYjRkNDU5MzUxMjdkM2Y1ZWRiZWM2)
- [basicModel_neutral_lbs_10_207_0_v1.0.0.pkl](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/hybrik/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl?versionId=CAEQHhiBgIC_v.zV6xciIDkwMDE4M2NjZTRkMjRmMWRiNTY3MWQ5YjQ0YzllNDYz)

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
    │   ├── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
    │   ├── h36m_mean_beta.npy
    │   ├── J_regressor_h36m.npy
    │   └── smpl
    │       ├── SMPL_FEMALE.pkl
    │       ├── SMPL_MALE.pkl
    │       └── SMPL_NEUTRAL.pkl
    ├── preprocessed_datasets
    │   ├── hybrik_coco_2017_train.npz
    │   ├── hybrik_h36m_train.npz
    │   └── hybrik_mpi_inf_3dhp_train.npz
    └── datasets
        ├── coco
        ├── h36m
        └── mpi_inf_3dhp
```


## Results and Models

We evaluate HybrIK on 3DPW. Values are MPJPE/PA-MPJPE.

| Config | 3DPW    | Download |
|:------:|:-------:|:------:|
| [resnet34_hybrik_mixed.py](resnet34_hybrik_mixed.py) | 81.08 / 49.02 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/hybrik/resnet34_hybrik_mixed-a61b3c9c_20220211.pth?versionId=CAEQKhiBgMDx0.Kd9xciIDA2NWFlMGVmNjNkMDQyYzE4NTFmMGJiYjczZWZmM2Rk) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/hybrik/20220121_170847.log?versionId=CAEQKhiBgICMgOyb9xciIDZkMTMyODYzMDc4NjRhZTliOThiM2JlMDE1MzhhYTBm) |
