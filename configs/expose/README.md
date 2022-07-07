#ExPose

## Introduction
We provide the config files for ExPose: [Monocular Expressive Body Regression through Body-Driven Attention](https://arxiv.org/abs/2008.09062).


```BibTeX
@inproceedings{ExPose:2020,
  title = {Monocular Expressive Body Regression through Body-Driven Attention},
  author = {Choutas, Vasileios and Pavlakos, Georgios and Bolkart, Timo and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages = {20--40},
  year = {2020},
  url = {https://expose.is.tue.mpg.de}
}
```

## Notes

- [SMPLX](https://smpl-x.is.tue.mpg.de/) v1.1 is used in our experiments.
- [FLAME](https://flame.is.tue.mpg.de/) 2019 is used in our experiments.
- [MANO](https://mano.is.tue.mpg.de/) v1.2 is used in our experiments.
- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used for body evaluation on 3DPW.
- [all_means.pkl]
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [MANO_SMPLX_vertex_ids.pkl](https://smpl-x.is.tue.mpg.de/)
- [shape_mean.npy]
- [SMPL-X__FLAME_vertex_ids.npy](https://smpl-x.is.tue.mpg.de/)
- [SMPLX_to_J14.npy]
- [flame_dynamic_embedding.npy]
- [flame_static_embedding.npy]
- [ExPose_curated_fits](https://expose.is.tue.mpg.de)
- [spin_in_smplx](https://expose.is.tue.mpg.de)
- [ffhq_annotations.npz]() We run [RingNet](https://ringnet.is.tue.mpg.de/) on FFHQ and then fitting to FAN 2D landmarks by [flame-fitting](https://github.com/HavenFeng/photometric_optimization).

As for pretrained model (hrnet_hmr_expose_body.pth). You can download it from [here]() and change the path of pretrained model in the config.
You can also pretrain the model using [hrnet_hmr_expose_body.py](hrnet_hmr_expose_body.py).

As for pretrained model (resnet18_hmr_expose_face.pth). You can download it from [here]() and change the path of pretrained model in the config.
You can also pretrain the model using [resnet18_hmr_expose_face.py](resnet18_hmr_expose_face.py).

As for pretrained model (resnet18_hmr_expose_hand.pth). You can download it from [here]() and change the path of pretrained model in the config.
You can also pretrain the model using [resnet18_hmr_expose_hand.py](resnet18_hmr_expose_hand.py).

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
    │   ├── all_means.pkl
    │   ├── J_regressor_h36m.npy
    │   ├── MANO_SMPLX_vertex_ids.pkl
    │   ├── shape_mean.npy
    │   ├── SMPL-X__FLAME_vertex_ids.npy
    │   ├── SMPLX_to_J14.npy
    │   ├── flame
    │   │   ├── FLAME_NEUTRAL.pkl
    │   │   ├── flame_dynamic_embedding.npy
    │   │   └── flame_static_embedding.npy
    │   ├── mano
    │   │   └── MANO_RIGHT.pkl
    │   ├── smpl
    │   │   ├── SMPL_FEMALE.pkl
    │   │   ├── SMPL_MALE.pkl
    │   │   └── SMPL_NEUTRAL.pkl
    │   └── smplx
    │       └── SMPLX_NEUTRAL.pkl
    ├── pretrained_models
    │   ├── hrnet_pretrain.pth
    │   ├── resnet18.pth
    │   ├── hrnet_hmr_expose_body.pth
    │   ├── resnet18_hmr_expose_face.pth
    │   └── resnet18_hmr_expose_hand.pth
    ├── preprocessed_datasets
    │   ├── curated_fits_train.npz
    │   ├── ehf_val.npz
    │   ├── ffhq_flame_train.npz
    │   ├── freihand_test.npz
    │   ├── freihand_train.npz
    │   ├── freihand_val.npz
    │   ├── h36m_train.npz
    │   ├── pw3d_test.npz
    │   ├── spin_smplx_train.npz
    │   └── stirling_ESRC3D_HQ.npz
    └── datasets
        ├── 3DPW
        ├── coco
        ├── EHF
        ├── ExPose_curated_fits
        │   └── train.npz
        ├── ffhq
        │   ├── ffhq_annotations.npz
        │   └── ffhq_global_images_1024
        ├── FreiHand
        ├── h36m
        ├── lsp
        │   ├── lsp_dataset_original
        │   └── lspet
        ├── mpii
        ├── spin_in_smplx
        │   ├── coco.npz
        │   ├── lsp.npz
        │   ├── lspet.npz
        │   └── mpii.npz
        └── stirling
```

## Results and Models

We evaluate hrnet_hmr_expose_body on 3DPW. Values are MPJPE/PA-MPJPE.

| Config | 3DPW | Download |
|:------:|:-------:|:------:|
| [hrnet_hmr_expose_body.py](hrnet_hmr_expose_body.py) | 92.59 / 60.43 | [model]() &#124; [log]() |


We evaluate resnet18_hmr_expose_face on Stirling/ESRC 3D. Values are 3DRMSE.
| Config | Stirling/ESRC 3D | Download |
|:------:|:-------:|:------:|
| [resnet18_hmr_expose_face.py](resnet18_hmr_expose_face.py) | 2.40 | [model]() &#124; [log]() |

We evaluate resnet18_hmr_expose_hand on FreiHand. Values are PA-MPJPE/PA-PVE.
| Config | FreiHand | Download |
|:------:|:-------:|:------:|
| [resnet18_hmr_expose_hand.py](resnet18_hmr_expose_hand.py) | 10.03 / 9.61 | [model]() &#124; [log]() |

We evaluate ExPose on EHF. Values are BODY PA-MPJPE/RIGHT_HAND PA-MPJPE/LEFT_HAND PA-MPJPE/PA-PVE/RIGHT_HAND PA-PVE/LEFT_HAND PA-PVE/FACE PA-PVE.
| Config | EHF | Download |
|:------:|:-------:|:------:|
| [expose.py](expose.py) | 55.70 / 14.6 / 14.4/ 56.65 / 14.6 / 14.5 / 6.90 | [model]() &#124; [log]() |