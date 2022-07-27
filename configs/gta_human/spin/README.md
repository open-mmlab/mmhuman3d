# GTA-Human (SPIN)

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
    - [gta_fits.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/data/static_fits/gta_fits.npy?versionId=CAEQRBiBgMCp4fPzjhgiIDhkZjhlODY2MjBkZjRjNzQ5ODZmNmVhY2IzNzA2ZmIy)
- Pretrained SPIN model [spin_official.pth](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/data/pretrained_models/spin_official.pth?versionId=CAEQRBiBgMC3zJPvjhgiIDNjODIxODJjYzEyNzRmNDhhNzU3Nzg3N2FlY2Y0ZWMx)

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
    │   └── spin_official.pth
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

| Config | 3DPW    | Download |
|:------:|:-------:|:------:|
| [resnet50_spin_gta_ft.py](resnet50_spin_gta_ft.py) | 83.20 / 51.98 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_spin_gta_ft/resnet50_spin_gta_ft-2538df81_20220708.pth?versionId=CAEQRBiBgICJxdjujhgiIGQwMTcwOGI5YzdlMTQ1ZjVhYzRhNWZkOTVhY2U3NjFm) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_spin_gta_ft/resnet50_spin_gta_ft.log?versionId=CAEQRBiBgMCHrdfujhgiIGRhZDA4NjY0NDBmNDRkMGRhMWRmODZlMzM1YmRiNzRj) |
