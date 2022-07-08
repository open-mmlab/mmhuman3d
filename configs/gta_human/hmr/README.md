# GTA-Human (HMR)

## Notes

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)
- [resnet50_hmr-2672873c_20220415.pth](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/hmr/resnet50_hmr-2672873c_20220415.pth?versionId=CAEQLxiBgIDuou6vgRgiIDFiOGRiZTA1ZTQ5NjRmMzdhYzkzY2ZmZGQwYjE0MzBl)

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
    │   ├── gta_human_4x.npz  
    │   └── pw3d_test.npz
    ├── pretrained
    │   └── resnet50_hmr-2672873c_20220415.pth
    └── datasets
        ├── coco
        ├── h36m
        ├── lspet
        ├── lsp
        ├── mpi_inf_3dhp
        ├── mpii
        ├── gta
        └── pw3d

```

## Results and Models

| Config | 3DPW    | Download |
|:------:|:-------:|:------:|
| [resnet50_hmr_gta_bt.py](resnet50_hmr_gta_bt.py) | 98.72 / 60.49 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_hmr_gta_bt-aa643b97_20220416.pth?versionId=CAEQLxiBgIDa4qHFgRgiIGUwNWJjZGFjMDE0OTRjYTg5MjI4MjcyZjI2YTVhMjli) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_hmr_gta_bt.log?versionId=CAEQLxiBgMCN4qHFgRgiIGZkNjJhMWY0YjFhODQxMGY5NTdmNjBhYTUwZDI3MmJj) |
| [resnet50_hmr_gta_ft.py](resnet50_hmr_gta_ft.py) | 91.42 / 55.71 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_hmr_gta_ft-f444e49c_20220416.pth?versionId=CAEQLxiBgMD04aHFgRgiIDg0YTExY2IzNWFmMjQ3MTc5NDFjY2MyNWU4MmM5Mzcz) &#124; [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gta_human/resnet50_hmr_gta_ft.log?versionId=CAEQLxiBgID936HFgRgiIDAwMDM5NDlkM2MyNzQxYTE4ZTgzZDc3ZGE4NTJlZTVh) |
