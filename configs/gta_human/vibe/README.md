# GTA-Human (VIBE)

## Notes

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)
- [gmm_08.pkl](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/gmm_08.pkl?versionId=CAEQHhiBgIDP6c3V6xciIGU4ZWFlYzlhNDJmODRmOGViYTMzOGRmODg2YjQ4NTg1)
- Pretrained SPIN model [spin_official.pth](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/data/pretrained_models/spin_official.pth?versionId=CAEQRBiBgMC3zJPvjhgiIDNjODIxODJjYzEyNzRmNDhhNzU3Nzg3N2FlY2Y0ZWMx) for extracting features. Rename it as spin.pth.

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
    │   └── spin.pth
    ├── preprocessed_datasets
    │   ├── vibe_insta_variety.npz
    │   ├── vibe_mpi_inf_3dhp_train.npz
    │   ├── vibe_pw3d_train.npz
    │   ├── vibe_pw3d_test.npz
    │   ├── vibe_gta_train.npz
    │   └── vibe_gta_96k.npz


```
