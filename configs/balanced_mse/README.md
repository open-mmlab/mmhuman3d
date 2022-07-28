# Balanced MSE

## Introduction

We provide the config files for **Balanced MSE for Imbalanced Visual Regression(CVPR2022 Oral)**.
[[Project]](https://github.com/jiawei-ren/BalancedMSE/) [[Paper]](https://arxiv.org/pdf/2203.16427.pdf)


```BibTeX
@article{BalancedMSE,
  author    = {Ren Jiawei and
               Zhang Mingyuan and
               Yu Cunjun and
               Liu Ziwei},
  title     = {Balanced MSE for Imbalanced Visual Regression},
  journal   = {arXiv preprint arXiv:2203.16427},
  year      = {2022}
}
```

## Notes

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)
- Pretrained model [spin_official_nofc.pth](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/balanced_mse/pretrained_models/spin_official_nofc.pth?versionId=CAEQSBiBgMCBm4STkhgiIDVmZGQwNTBmNTU1YjQyYjRhMzJjNzM1Mjc1OTgwNzJj)
- Static fits required for the training:
    - [coco_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/coco_fits.npy?versionId=CAEQHhiBgMCr4ZvV6xciIGY1OTZjM2NlZWI3ZDRjMzI5ODE0MWQxYjM2M2Y4NTVk)
    - [h36m_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/h36m_fits.npy?versionId=CAEQHhiBgIC54ZvV6xciIDc2YjExNmM0NjBiMDQwMmU5NjJmODljNjgxYWE1MGQx)
    - [lspet_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/lspet_fits.npy?versionId=CAEQHhiBgIDy4ZvV6xciIDkyMjQ3OGM2YWU5YTRlNTI4MmYxM2I5Njg1Yzc3OWYw)
    - [lsp_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/lsp_fits.npy?versionId=CAEQHhiBgIDS4ZvV6xciIGFmMzdhMWJjZWQ1MjRkODBiZDY3NGU0MTc1Yzg0Nzlh)
    - [mpii_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/mpii_fits.npy?versionId=CAEQHhiBgMCm4ZvV6xciIGM1OTIzZDlkNjVhODQ2MDY5ODkyZWE4ZDEzZGJlNTdi)
    - [mpi_inf_3dhp_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/spin/static_fits/mpi_inf_3dhp_fits.npy?versionId=CAEQHhiBgMDf4ZvV6xciIDQyYjRmMzdhODdmNDQ1YTBhYzY4MTk1OTAxNzc4MmVj)

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
    ├── pretrained_models
    │   └── spin_official_nofc.pth
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

We evaluate methods on 3DPW with new metrics: bMPVPE, bMPJPE, bPA-MPJPE. For each metric, we provide three values: average error of 100 bins (All), average error of tail 10% data (Tail 10%), and average error of tail 5% data (Tail 5%). The definition of these metrics can be found in Section 4.2.2 of [paper](https://arxiv.org/pdf/2203.16427.pdf). The below tables report results in form of "All / Tail 10% / Tail 5%".

| Config |  bMPVPE | bMPJPE | bPA-MPJPE | Download |
|:------:|:-------:|:------:|:------:|:------:|
| [resnet50_spin_ihmr_ft_baseline.py](resnet50_spin_ihmr_ft_baseline.py) | 115.33/126.98/132.20 | 98.69/111.79/113.61 | 65.98/76.89/76.76 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/balanced_mse/resnet50_spin_ihmr_ft_baseline/resnet50_spin_ihmr_ft_baseline.pth?versionId=CAEQSBiBgMDpm4STkhgiIDU0ODBlNjUzMjk1MjQ2MjQ5MDgxNzBlZjZjNWQyODVl) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/balanced_mse/resnet50_spin_ihmr_ft_baseline/resnet50_spin_ihmr_ft_baseline.log?versionId=CAEQSBiBgMCzwYGTkhgiIDFlMDVkMDU5M2UxYzQwMDA4NTNjODgzMmRiNzNmOGYy) |
| [resnet50_spin_ihmr_ft_bmc.py](resnet50_spin_ihmr_ft_bmc.py) | 112.67/126.13/130.42 | 95.70/110.22/111.62 | 65.65/75.61/75.86 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/balanced_mse/resnet50_spin_ihmr_ft_bmc/resnet50_spin_ihmr_ft_bmc.pth?versionId=CAEQSBiBgMCK8IKTkhgiIGRlNzA2MGYwMmM4YjRhMzE4NjA2MmNmYzAwMmM0ZmRj) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/balanced_mse/resnet50_spin_ihmr_ft_bmc/resnet50_spin_ihmr_ft_bmc.log?versionId=CAEQSBiBgICSv4GTkhgiIDljMzdjMGQ3MGJjNDQ4OTJiN2M2YjI4OTI2ZjQ3ZTc3) |
