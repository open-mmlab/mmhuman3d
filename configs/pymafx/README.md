# PyMAF-X

## Introduction

We provide the config files for [PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images](https://arxiv.org/abs/2207.06400).

```BibTeX
@inproceedings{pymaf2021,
  title={PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop},
  author={Zhang, Hongwen and Tian, Yating and Zhou, Xinchi and Ouyang, Wanli and Liu, Yebin and Wang, Limin and Sun, Zhenan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2021}
}

@article{pymafx2022,
  title={PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images},
  author={Zhang, Hongwen and Tian, Yating and Zhang, Yuxiang and Li, Mengcheng and An, Liang and Sun, Zhenan and Liu, Yebin},
  journal={arXiv preprint arXiv:2207.06400},
  year={2022}
}
```

## Notes

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
  - Neutral model can be downloaded from [SMPLify](https://smplify.is.tue.mpg.de/).
  - All body models have to be renamed in `SMPL_{GENDER}.pkl` format. <br/>
    For example, `mv basicModel_neutral_lbs_10_207_0_v1.0.0.pkl SMPL_NEUTRAL.pkl`
- [SMPLX](https://smpl-x.is.tue.mpg.de/) v1.1 is used in our experiments.
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)
- Download [smpl_downsampling.npz](https://github.com/nkolot/GraphCMR/raw/master/data/mesh_downsampling.npz) from `nkolot/GraphCMR`.
- Download [mano_downsampling.npz](https://github.com/microsoft/MeshGraphormer/raw/main/src/modeling/data/mano_downsampling.npz) from `microsoft/MeshGraphormer`.
- Download the pre-trained [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pymaf_x/PyMAF-X_model_checkpoint.pth).
- Download the `partial_mesh` files from [PyMAF-X](https://cloud.tsinghua.edu.cn/d/3bc20811a93b488b99a9/) or use the following script: <br/>

  ```bash
  mkdir mmhuman3d_download
  cd mmhuman3d_download
  wget -O mmhuman3d.7z -q https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/mmhuman3d.7z
  7za x mmhuman3d.7z
  cp -r mmhuman3d/data/partial_mesh/ ../data/
  cd ..
  rm -rf mmhuman3d_download
  ```

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
    │   ├── smpl_mean_params.npz
    │   ├── smpl
    │   │   ├── SMPL_FEMALE.pkl
    │   │   ├── SMPL_MALE.pkl
    │   │   └── SMPL_NEUTRAL.pkl
    │   └── smplx
    │       ├── smplx_to_smpl.npz
    │       └── SMPLX_NEUTRAL.npz
    ├── partial_mesh
    │   └── *_vids.npz.npz
    ├── pretrained_models
    │   └── PyMAF-X_model_checkpoint.pth
    ├── mano_downsampling.npz
    └── smpl_downsampling.npz

```

## Demo

By default, we use mmpose to detect 2d keypoints, and you can get the SMPL-X parameters as follow:

```bash
python demo/pymafx_estimate_smplx.py \
    --input_path demo/resources/multi_person_demo.mp4 \
    --output_path output \
    --visualization
```
If you want to reproduce the original repos, please install openpifpaf,
then you will get the SMPL-X parameters as follow:

```bash
python demo/pymafx_estimate_smplx.py \
    --input_path demo/resources/multi_person_demo.mp4 \
    --output_path output \
    --visualization \
    --use_openpifpaf
```

You can find results in `output`.
