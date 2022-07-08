# GTA-Human (PARE)

## Notes

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)
- Pascal Occluders for the pretraining:
  - [pascal_occluders.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/pascal_occluders.npy?versionId=CAEQOhiBgMCH2fqigxgiIDY0YzRiNThkMjU1MzRjZTliMTBhZmFmYWY0MTViMTIx)
- Pretrained PARE model [hrnet_w32_conv_pare_mosh.pth]()


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
    │   └── hrnet_w32_conv_pare_mosh.pth
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

| Config | 3DPW    | Download |
|:------:|:-------:|:------:|
| [hrnet_w32_conv_pare_gta_ft.py](hrnet_w32_conv_pare_gta_ft.py) | 77.52 / 46.84 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/hrnet_w32_conv_pare_gta_ft/hrnet_w32_conv_pare_gta_ft-838829bc_20220708.pth?versionId=CAEQRBiBgMDRxNjujhgiIGY3ZmUzMjUzZjJhNjQ2MTg5ODNjMWFlNTJmMGJhMmFh) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/hrnet_w32_conv_pare_gta_ft/hrnet_w32_conv_pare_gta_ft.log?versionId=CAEQRBiBgICjxdfujhgiIGZiZDFmMmI1YWI0MzQyZjM4MmQ2MjZiYzY5OGQ5ODk1) |
