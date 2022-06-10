# 数据预处理

<!-- - [Data preparation](#data-preparation) -->
  <!-- * [Overview](#overview)
  * [Generate dataset files](#generate-dataset-files)
  * [Obtain preprocessed datasets](#obtain-preprocessed-datasets) -->
- [不同算法使用的数据集](#datasets-for-supported-algorithms)
- [文件夹结构](#folder-structure)
  * [AGORA](#agora)
  * [COCO](#coco)
  * [COCO-WholeBody](#coco-wholebody)
  * [CrowdPose](#crowdpose)
  * [EFT](#eft)
  * [GTA-Human](#gta-human)
  * [Human3.6M](#human36m)
  * [Human3.6M Mosh](#human36m-mosh)
  * [HybrIK](#hybrik)
  * [LSP](#lsp)
  * [LSPET](#lspet)
  * [MPI-INF-3DHP](#mpi-inf-3dhp)
  * [MPII](#mpii)
  * [PoseTrack18](#posetrack18)
  * [Penn Action](#penn-action)
  * [PW3D](#pw3d)
  * [SPIN](#spin)
  * [SURREAL](#surreal)


## 总览

我们使用 [HumanData](./human_data.md) 结构用于存储和加载数据。经过处理的.npz可以使用我们提供的数据转换脚本从原始数据格式获取，详情请参考[convert_datasets.py](https://github.com/open-mmlab/mmhuman3d/tree/main/tools/convert_datasets.py).

如下是我们支持的格式转换方式和具体的 `数据集名称`:
- AgoraConverter (`agora`)
- AmassConverter (`amass`)
- CocoConverter (`coco`)
- CocoHybrIKConverter (`coco_hybrik`)
- CocoWholebodyConverter (`coco_wholebody`)
- CrowdposeConverter (`crowdpose`)
- EftConverter (`eft`)
- GTAHumanConverter (`gta_human`)
- H36mConverter (`h36m_p1`, `h36m_p2`)
- H36mHybrIKConverter (`h36m_hybrik`)
- H36mSpinConverter (`h36m_spin`)
- InstaVibeConverter (`instavariety_vibe`)
- LspExtendedConverter (`lsp_extended`)
- LspConverter (`lsp_original`, `lsp_dataset`)
- MpiiConverter (`mpii`)
- MpiInf3dhpConverter (`mpi_inf_3dhp`)
- MpiInf3dhpHybrIKConverter (`mpi_inf_3dhp_hybrik`)
- PennActionConverter (`penn_action`)
- PosetrackConverter (`posetrack`)
- Pw3dConverter (`pw3d`)
- Pw3dHybrIKConverter (`pw3d_hybrik`)
- SurrealConverter (`surreal`)
- SpinConverter (`spin`)
- Up3dConverter (`up3d`)

<!--

### Generate dataset files

After preparing the data the datasets according to their respective folder structure (see below), simply run this command to get the preprocessed npz file from your specified `dataset-name`
in $YOUR_OUTPUT_PATH:

```bash
python tools/convert_datasets.py \
  --datasets dataset-name \
  --root_path $YOUR_ROOT_PATH \
  --output_path $YOUR_OUTPUT_PATH
```

Use `all` to preprocess all datasets in the supported configs:
```bash
python tools/convert_datasets.py \
  --datasets all \
  --root_path $YOUR_ROOT_PATH \
  --output_path $YOUR_OUTPUT_PATH
```

### Obtain preprocessed datasets

The available dataset configurations are listed [here](https://github.com/open-mmlab/mmhuman3d/tree/main/tools/convert_datasets.py).

An example is
```
DATASET_CONFIGS = dict(
    ...
    pw3d=dict(type='Pw3dConverter', modes=['train', 'test'], prefix='pw3d')
)
```

where `pw3d` is an example of a `dataset-name`. The available modes are `train` and `test` and the prefix specifies the name of the dataset folder. In this case, `pw3d` is the name
of the dataset folder containing the raw annotations and images arranged in the following [structure](#pw3d).

Running this command

```bash
python tools/convert_datasets.py \
  --datasets pw3d \
  --root_path data/datasets \
  --output_path data/preprocessed_datasets
```

would allow us to obtain the preprocessed npz files under `data/preprocessed_datasets`:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── datasets
    └── preprocessed_datasets
        ├── pw3d_train.npz
        └── pw3d_test.npz
```

We can also modify the mode in the dataset configuration if we only want to obtain the
preprocessed npz file in `test` mode. i.e.
```
DATASET_CONFIGS = dict(
    ...
    pw3d=dict(type='Pw3dConverter', modes=['test'], prefix='pw3d')
)
```

-->

## 不同算法使用的数据集

所有算法使用的数据集路径和经过处理的.npz文件路径为`data/datasets` 和 `data/preprocessed_datasets`。使用如下命令进行数据格式转换：

```bash
python tools/convert_datasets.py \
  --datasets <dataset-name> \
  --root_path data/datasets \
  --output_path data/preprocessed_datasets
```

使用时，请指定具体的`dataset-name`.

训练HMR算法，需要如下的数据集
  - [COCO](#coco)
  - [Human3.6M](#human36m)
  - [Human3.6M Mosh](#human36m-mosh)
  - [MPI-INF-3DHP](#mpi-inf-3dhp)
  - [MPII](#mpii)
  - [LSP](#lsp)
  - [LSPET](#lspet)
  - [PW3D](#pw3d)

使用如下的数据集名称替换`dataset-names`进行数据转换:
```
coco, pw3d, mpii, mpi_inf_3dhp, lsp_original, lsp_extended, h36m
```

**或者**, 您可以下载处理好的.npz文件:
- [cmu_mosh.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/cmu_mosh.npz?versionId=CAEQHhiBgIDoof_37BciIDU0OGU0MGNhMjAxMjRiZWI5YzdkMWEzMzc3YzBiZDM2)
- [coco_2014_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/coco_2014_train.npz?versionId=CAEQHhiBgICUrvbS6xciIDFmZmFhMDk5OGQ3YzQ5ZDE5NzJkMGQxNzdmMmQzZDdi)
- [h36m_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/h36m_train.npz?versionId=CAEQHhiBgMDrrfbS6xciIGY2NjMxMjgwMWQzNjRkNWJhYTNkZTYyYWUxNWQ4ZTE5)
- [lsp_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/lsp_train.npz?versionId=CAEQHhiBgICnq_bS6xciIDU4ZTRhMDIwZTBkZjQ1YTliYTY0NGFmMDVmOGVhZjMy)
- [lspet_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/lspet_train.npz?versionId=CAEQHhiBgICXrPbS6xciIDVkZGNmYWZjODlmMzQ2YjNhMjhlNmJmMzU2MjM4Yzg4)
- [mpi_inf_3dhp_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/mpi_inf_3dhp_train.npz?versionId=CAEQHhiBgMD3q_bS6xciIGQwYjc4NTRjYTllMzRkODU5NTNiZDQyOTBlYmRhODg5)
- [mpii_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/mpii_train.npz?versionId=CAEQHhiBgIDhq_bS6xciIDEwMmE0ZDc0NWI1NjQ2NWZhYTA5ZjEyODBiNWFmODg1)
- [pw3d_test.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/pw3d_test.npz?versionId=CAEQHhiBgMDaq_bS6xciIGVjY2YzZGJkNjNmMjQ2NGU4OTZkYjMwMjhhYWM1Y2I0)

由于许可证的限制，我们无法上传`h36m_mosh_train.npz`。但是我们提供了相关的转换工具, 如果您拥有原始的`mosh`数据, 您可以参考[Human3.6M Mosh](#human36m-mosh)。

处理好的数据集应该具有如下的结构:
```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── datasets
    └── preprocessed_datasets
        ├── coco_2014_train.npz
        ├── h36m_train.npz (h36m_mosh_train.npz)
        ├── lspet_train.npz
        ├── lsp_train.npz
        ├── mpi_inf_3dhp_train.npz
        ├── mpii_train.npz
        └── pw3d_test.npz
```

训练SPIN算法, 需要如下的数据集:
  - [COCO](#coco)
  - [Human3.6M](#human36m)
  - [Human3.6M Mosh](#human36m-mosh)
  - [MPI-INF-3DHP](#mpi-inf-3dhp)
  - [MPII](#mpii)
  - [LSP](#lsp)
  - [LSPET](#lspet)
  - [PW3D](#pw3d)
  - [SPIN](#spin)


使用如下的数据集名称替换`dataset-names`进行数据转换:
```
spin, h36m
```

**或者**, 您可以先下载处理好的.npz文件:
- [spin_coco_2014_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/spin_coco_2014_train.npz?versionId=CAEQHhiBgICb6bfT6xciIGM2NmNmZDYyNDMxMDRiNTVhNDk3YzY1N2Y2ODdlMTAy)
- [h36m_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/h36m_train.npz?versionId=CAEQHhiBgMDrrfbS6xciIGY2NjMxMjgwMWQzNjRkNWJhYTNkZTYyYWUxNWQ4ZTE5)
- [spin_lsp_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/spin_lsp_train.npz?versionId=CAEQHhiBgIDu57fT6xciIDQ0ODAzNjUyNjJkMzQyNzQ5Y2IzNGNhOTZmZGI2NzBm)
- [spin_lspet_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/spin_lspet_train.npz?versionId=CAEQHhiBgMCe6LfT6xciIDc3NzZiYzA1ZGJkYzQwNzRhYjg3ZDMwYTdjZDZmNTAw)
- [spin_mpi_inf_3dhp_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/spin_mpi_inf_3dhp_train.npz?versionId=CAEQHhiBgMCV6LfT6xciIDliYTJhM2FkNDkyYjRiOWFiYTUwOTk0MGRlNThlZWRk)
- [spin_mpii_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/spin_mpii_train.npz?versionId=CAEQHhiBgMDz57fT6xciIGJjMzAwMDdlYTBmMTQ0MDg4ZGE4YjhiZGNkNWQwZmM1)
- [spin_pw3d_test.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/spin_pw3d_test.npz?versionId=CAEQHhiBgMCL6LfT6xciIGUxNjY3OTBiODU5ZDQxODliYTQ4NzU0OGVjMzJkYmRm)

由于许可证的限制，我们无法上传`h36m_mosh_train.npz`。但是我们提供了相关的转换工具, 如果您拥有原始的`mosh`数据, 您可以参考[Human3.6M Mosh](#human36m-mosh)。

处理好的数据集应该具有如下的结构:
```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── datasets
    └── preprocessed_datasets
        ├── spin_coco_2014_train.npz
        ├── h36m_train.npz (h36m_mosh_train.npz)
        ├── spin_lsp_train.npz
        ├── spin_lspet_train.npz
        ├── spin_mpi_inf_3dhp_train.npz
        ├── spin_mpii_train.npz
        └── spin_pw3d_test.npz
```

训练VIBE算法, 需要如下的数据集:
  - [MPI-INF-3DHP](#mpi-inf-3dhp)
  - [PW3D](#pw3d)

数据转换暂时不可用.

**或者**, 您可以先下载处理好的.npz文件:
- [vibe_insta_variety.npz](https://pjlab-my.sharepoint.cn/:u:/g/personal/openmmlab_pjlab_org_cn/EYnlkp-69NBNlXDH-5ELZikBXDbSg8SZHqmdSX_3hK4EYg?e=QUl5nI)
- [vibe_mpi_inf_3dhp_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/vibe_mpi_inf_3dhp_train.npz?versionId=CAEQHhiBgICTnq3U6xciIGUwMTc5YWQ2MjNhZDQ3NGE5MmYxOWJhMGQxMTcwNTll)
- [vibe_pw3d_test.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/vibe_pw3d_test.npz?versionId=CAEQHhiBgMD5na3U6xciIGQ4MmU0MjczYTYzODQ1NDQ5M2JiNzY1N2E5MTNlOWY5)


处理好的数据集应该具有如下的结构:
```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── datasets
    └── preprocessed_datasets
        ├── vibe_insta_variety.npz
        ├── vibe_mpi_inf_3dhp_train.npz
        └── vibe_pw3d_test.npz
```

训练HybrIK算法, 需要如下的数据集:
  - [HybrIK](#hybrik)
  - [COCO](#coco)
  - [Human3.6M](#human36m)
  - [MPI-INF-3DHP](#mpi-inf-3dhp)
  - [PW3D](#pw3d)

使用如下的数据集名称替换`dataset-names`进行数据转换:
```
h36m_hybrik, pw3d_hybrik, mpi_inf_3dhp_hybrik, coco_hybrik
```

**或者**, 您可以先下载处理好的.npz文件:
- [hybriK_coco_2017_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/hybrik_coco_2017_train.npz?versionId=CAEQHhiBgMDA6rjT6xciIDE3N2FiZDkxYTkyZDRjN2ZiYjc1ODQ2YTc5NjY0ZmFl)
- [hybrik_h36m_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/hybrik_h36m_train.npz?versionId=CAEQHhiBgIC_iLjT6xciIGE4NmQ5YzUxMzY0ZjQ0Y2U5MWFkOTkwNmIwMGI4NTNm)
- [hybrik_mpi_inf_3dhp_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/hybrik_mpi_inf_3dhp_train.npz?versionId=CAEQHhiBgICogLjT6xciIDQwYzRlYTVlOTE0YTQ4ZDRhYTljOGRkZDc1MDhjNDgy)
- [hybrik_pw3d_test.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/hybrik_pw3d_test.npz?versionId=CAEQHhiBgMCO8LfT6xciIDhjMDFhOTFmZjY4MDQ4MWI4MzVmODYyYTc1NTYwNjA1)


处理好的数据集应该具有如下的结构:
```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── datasets
    └── preprocessed_datasets
        ├── hybriK_coco_2017_train.npz
        ├── hybrik_h36m_train.npz
        ├── hybrik_mpi_inf_3dhp_train.npz
        └── hybrik_pw3d_test.npz
```

训练PARE算法, 需要如下的数据集:
  - [Human3.6M](#human36m)
  - [Human3.6M Mosh](#human36m-mosh)
  - [MPI-INF-3DHP](#mpi-inf-3dhp)
  - [EFT-COCO](#EFT)
  - [EFT-MPII](#EFT)
  - [EFT-LSPET](#EFT)
  - [PW3D](#pw3d)


使用如下的数据集名称替换`dataset-names`进行数据转换:
```
h36m, coco, mpii, lspet, mpi-inf-3dhp, pw3d
```

**或者**, 您可以先下载处理好的.npz文件:
- [h36m_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/h36m_train.npz?versionId=CAEQHhiBgMDrrfbS6xciIGY2NjMxMjgwMWQzNjRkNWJhYTNkZTYyYWUxNWQ4ZTE5)
- [mpi_inf_3dhp_train.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/mpi_inf_3dhp_train.npz?versionId=CAEQHhiBgMD3q_bS6xciIGQwYjc4NTRjYTllMzRkODU5NTNiZDQyOTBlYmRhODg5)
- [eft_mpii.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/eft_mpii.npz?versionId=CAEQOhiBgMCXlty_gxgiIDYxNDc5YTIzZjBjMDRhMGM5ZjBiZmYzYjFjMTU1ZTRm)
- [eft_lspet.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/eft_lspet.npz?versionId=CAEQOhiBgMC339u_gxgiIDZlNzk1YjMxMWRmMzRkNWJiNjg1OTI2Mjg5OTA1YzJh
)
- [eft_coco_all.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/eft_coco_all.npz?versionId=CAEQOhiBgID3iuS_gxgiIDgwYzU4NTc3ZWRkNDQyNGJiYzU4MGViYTFhYTFmMmUx)
- [pw3d_test.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/pw3d_test.npz?versionId=CAEQHhiBgMDaq_bS6xciIGVjY2YzZGJkNjNmMjQ2NGU4OTZkYjMwMjhhYWM1Y2I0)


处理好的数据集应该具有如下的结构:
```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── datasets
    └── preprocessed_datasets
        ├── h36m_mosh_train.npz
        ├── h36m_train.npz
        ├── mpi_inf_3dhp_train.npz
        ├── eft_mpii.npz
        ├── eft_lspet.npz
        ├── eft_coco_all.npz
        └── pw3d_test.npz
```

## 文件夹结构

### AGORA

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/pdf/2104.14643.pdf">AGORA (CVPR'2021)</a></summary>

```bibtex
@inproceedings{Patel:CVPR:2021,
 title = {{AGORA}: Avatars in Geography Optimized for Regression Analysis},
 author = {Patel, Priyanka and Huang, Chun-Hao P. and Tesch, Joachim and Hoffmann, David T. and Tripathi, Shashank and Black, Michael J.},
 booktitle = {Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition ({CVPR})},
 month = jun,
 year = {2021},
 month_numeric = {6}
}
```

</details>

请从[这里](https://agora.is.tue.mpg.de/index.html)下载AGORA数据集，并按照如下结构放置文件夹:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
   └── datasets
       └── agora
           ├── camera_dataframe # smplx annotations
           │   ├── train_0_withjv.pkl
           │   ├── validation_0_withjv.pkl
           │   └── ...
           ├── camera_dataframe_smpl # smpl annotations
           │   ├── train_0_withjv.pkl
           │   ├── validation_0_withjv.pkl
           │   └── ...
           ├── images
           │   ├── train
           │   │   ├── ag_trainset_3dpeople_bfh_archviz_5_10_cam00_00000_1280x720.png
           │   │   ├── ag_trainset_3dpeople_bfh_archviz_5_10_cam00_00001_1280x720.png
           │   │   └── ...
           │   ├── validation
           │   └── test
           ├── smpl_gt
           │   ├── trainset_3dpeople_adults_bfh
           │   │   ├── 10004_w_Amaya_0_0.mtl
           │   │   ├── 10004_w_Amaya_0_0.obj
           │   │   ├── 10004_w_Amaya_0_0.pkl
           │   │   └── ...
           │   └── ...
           └── smplx_gt
                   ├── 10004_w_Amaya_0_0.obj
                   ├── 10004_w_Amaya_0_0.pkl
                   └── ...
```

### AMASS

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://files.is.tue.mpg.de/black/papers/amass.pdf">AMASS (ICCV'2019)</a></summary>

```bibtex
@inproceedings{AMASS:2019,
  title={AMASS: Archive of Motion Capture as Surface Shapes},
  author={Mahmood, Naureen and Ghorbani, Nima and F. Troje, Nikolaus and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year={2019},
  month = {Oct},
  url = {https://amass.is.tue.mpg.de},
  month_numeric = {10}
}
```

</details>

未来会添加更多关于处理AMASS数据集的细节。

**或者**, 您可以直接下载处理好的.npz文件:
- [amass_smplh.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/amass_smplh.npz?versionId=CAEQIhiBgICS4Mrt7xciIGU5MDBmZmE4Y2I0NjRiYTc4ZWY2NzY2MzU1ZmIwZTQ2)
- [amass_smplx.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/datasets/amass_smplx.npz?versionId=CAEQIhiBgIDh387t7xciIGRlN2JlZjA0ZGM0YzRkNmM5OWJhNmVjMmZlN2RiN2E1)

### COCO

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48">COCO (ECCV'2014)</a></summary>

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

</details>

请从[这里](http://cocodataset.org/#download)下载`COCO`数据集。其中，训练HMR需要`COCO2014`，训练HybrIK需要`COCO2017`。
下载并解压至  `$MMHUMAN3D/data/datasets`, 使它们具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── coco
            ├── annotations
            |   ├── person_keypoints_train2014.json
            |   ├── person_keypoints_val2014.json
            ├── train2014
            │   ├── COCO_train2014_000000000009.jpg
            │   ├── COCO_train2014_000000000025.jpg
            │   ├── COCO_train2014_000000000030.jpg
            |   └── ...
            └── train_2017
                │── annotations
                │   ├── person_keypoints_train2017.json
                │   └── person_keypoints_val2017.json
                │── train2017
                │   ├── 000000000009.jpg
                │   ├── 000000000025.jpg
                │   ├── 000000000030.jpg
                │   └── ...
                └── val2017
                    ├── 000000000139.jpg
                    ├── 000000000285.jpg
                    ├── 000000000632.jpg
                    └── ...
```


### COCO-WholeBody

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

请从[这里](http://cocodataset.org/#download)下载[COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody/)数据集(2017 train/val)中的图像，从[这里](https://github.com/jin-s13/COCO-WholeBody/)下载标注文件。
解压至`$MMHUMAN3D/data/datasets`文件夹，并使其具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── coco
            ├── annotations
            |   ├── coco_wholebody_train_v1.0.json
            |   └── coco_wholebody_val_v1.0.json
            └── train_2017
                │── train2017
                │   ├── 000000000009.jpg
                │   ├── 000000000025.jpg
                │   ├── 000000000030.jpg
                │   └── ...
                └── val2017
                    ├── 000000000139.jpg
                    ├── 000000000285.jpg
                    ├── 000000000632.jpg
                    └── ...

```




### CrowdPose

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Li_CrowdPose_Efficient_Crowded_Scenes_Pose_Estimation_and_a_New_Benchmark_CVPR_2019_paper.html">CrowdPose (CVPR'2019)</a></summary>

```bibtex
@article{li2018crowdpose,
  title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
  author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
  journal={arXiv preprint arXiv:1812.00324},
  year={2018}
}
```

</details>

下载[CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)数据集，并解压至文件夹`$MMHUMAN3D/data/datasets`，使其具有如下结构：

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── crowdpose
            ├── crowdpose_train.json
            ├── crowdpose_val.json
            ├── crowdpose_trainval.json
            ├── crowdpose_test.json
            └── images
                ├── 100000.jpg
                ├── 100001.jpg
                ├── 100002.jpg
                └── ...
```

### EFT

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/pdf/2004.03686.pdf">EFT (3DV'2021)</a></summary>

```bibtex
@inproceedings{joo2020eft,
 title={Exemplar Fine-Tuning for 3D Human Pose Fitting Towards In-the-Wild 3D Human Pose Estimation},
 author={Joo, Hanbyul and Neverova, Natalia and Vedaldi, Andrea},
 booktitle={3DV},
 year={2020}
}
```

</details>

下载[EFT](https://github.com/facebookresearch/eft)数据集，解压至`$MMHUMAN3D/data/datasets`，并使其具有如下结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
   └── datasets
       └── eft
           ├── coco_2014_train_fit
           |   ├── COCO2014-All-ver01.json
           |   └── COCO2014-Part-ver01.json
           |── LSPet_fit
           |   └── LSPet_ver01.json
           └── MPII_fit
               └── MPII_ver01.json
```

### GTA-Human

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://gta-human.com/">GTA-Human (arXiv'2021)</a></summary>

```bibtex
@article{cai2021playing,
  title={Playing for 3D Human Recovery},
  author={Cai, Zhongang and Zhang, Mingyuan and Ren, Jiawei and Wei, Chen and Ren, Daxuan and Li, Jiatong and Lin, Zhengyu and Zhao, Haiyu and Yi, Shuai and Yang, Lei and others},
  journal={arXiv preprint arXiv:2110.07588},
  year={2021}
}
```

More details are coming soon!


</details>

### Human3.6M

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6682899/">Human3.6M (TPAMI'2014)</a></summary>

```bibtex
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,  Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {IEEE Computer Society},
  volume = {36},
  number = {7},
  pages = {1325-1339},
  month = {jul},
  year = {2014}
}
```

</details>

从官网下载[Human3.6M](http://vision.imar.ro/human3.6m/description.php)数据集，并使用[脚本](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/data/data_converters/h36m.py)进行预处理，间隔5帧提取标注。
处理过的数据集具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── h36m
            ├── annot
            ├── S1
            |   ├── images
            |   |    |── S1_Directions_1.54138969
            |   |    |  ├── S1_Directions_1.54138969_00001.jpg
            |   |    |  ├── S1_Directions_1.54138969_00002.jpg
            |   |    |  └── ...
            |   |    └── ...
            |   ├── MyPoseFeatures
            |   |    |── D2Positions
            |   |    └── D3_Positions_Mono
            |   ├── MySegmentsMat
            |   |    └── ground_truth_bs
            |   └── Videos
            |        |── Directions 1.54138969.mp4
            |        |── Directions 1.55011271.mp4
            |        └── ...
            ├── S5
            ├── S6
            ├── S7
            ├── S8
            ├── S9
            ├── S11
            └── metadata.xml
```

修改[配置文件](https://github.com/open-mmlab/mmhuman3d/blob/main/tools/convert_datasets.py)中的`h36m_p1`，从原始的[Human3.6M](http://vision.imar.ro/human3.6m/description.php)数据集中的视频提取图像。

```python
h36m_p1=dict(
    type='H36mConverter',
    modes=['train', 'valid'],
    protocol=1,
    extract_img=True, # 从原视频中提取图像，设置为True
    prefix='h36m'),
```

### Human3.6M Mosh

<!-- [DATASET] -->

我们使用[HMR](https://github.com/akanazawa/hmr)提供的[MoShed](https://mosh.is.tue.mpg.de/)数据集训练`HMR`、`SPIN`和`PARE`。由于版权的限制，我们无法上传该数据集。即使没有使用mosh参数的许可，仍可使用我们提供的[转换脚本](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/data/data_converters/h36m.py)生成h36m的.npz文件。

修改[配置文件](https://github.com/open-mmlab/mmhuman3d/blob/main/tools/convert_datasets.py)中的`h36m_p1`，从原始的[Human3.6M](http://vision.imar.ro/human3.6m/description.php)数据集中的视频提取图像。

不具有mosh数据的配置文件:
```python
h36m_p1=dict(
    type='H36mConverter',
    modes=['train', 'valid'],
    protocol=1,
    extract_img=True,  # 从原视频中提取图像，设置为True
    prefix='h36m'),
```

具有mosh数据的配置文件:
```python
h36m_p1=dict(
    type='H36mConverter',
    modes=['train', 'valid'],
    protocol=1,
    extract_img=True,  # 从原视频中提取图像，设置为True
    mosh_dir='data/datasets/h36m_mosh', # 如果拥有mosh数据，指定其的路径
    prefix='h36m'),
```

如果您可以获取到Human3.6m的Mosh数据，整个文件夹应该具有如下的架构：

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── h36m_mosh
            ├── annot
            ├── S1
            |   ├── images
            |   |    ├── Directions 1_cam0_aligned.pkl
            |   |    ├── Directions 1_cam1_aligned.pkl
            |   |    └── ...
            ├── S5
            ├── S6
            ├── S7
            ├── S8
            ├── S9
            └── S11
```

### HybrIK

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/CVPR2021/html/Li_HybrIK_A_Hybrid_Analytical-Neural_Inverse_Kinematics_Solution_for_3D_Human_CVPR_2021_paper.html">HybrIK (CVPR'2021)</a></summary>

```bibtex
@inproceedings{li2020hybrikg,
  author = {Li, Jiefeng and Xu, Chao and Chen, Zhicun and Bian, Siyuan and Yang, Lixin and Lu, Cewu},
  title = {HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation},
  booktitle={CVPR 2021},
  pages={3383--3393},
  year={2021},
  organization={IEEE}
}
```

</details>

下载[HybrIK](https://github.com/Jeff-sjtu/HybrIK)数据集的[标注](https://github.com/Jeff-sjtu/HybrIK#fetch-data), 并使其文件夹具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── hybrik_data
            ├── Sample_5_train_Human36M_smpl_leaf_twist_protocol_2.json
            ├── Sample_20_test_Human36M_smpl_protocol_2.json
            ├── 3DPW_test_new.json
            ├── annotation_mpi_inf_3dhp_train_v2.json
            └── annotation_mpi_inf_3dhp_test.json
```

运行如下脚本，将.json文件处理成mmhuman3d中使用的.npz文件：

  - [Human3.6M](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/data/data_converters/h36m_hybrik.py)
  - [PW3D](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/data/data_converters/pw3d_hybrik.py)
  - [Mpi-Inf-3dhp](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/data/data_converters/mpi_inf_3dhp_hybrik.py)
  - [COCO](https://github.com/open-mmlab/mmhuman3d/tree/main/mmhuman3d/data/data_converters/coco_hybrik.py)


### LSP

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://sam.johnson.io/research/publications/johnson10bmvc.pdf">LSP (BMVC'2010)</a></summary>

```bibtex
@inproceedings{johnson2010clustered,
  title={Clustered Pose and Nonlinear Appearance Models for Human Pose Estimation.},
  author={Johnson, Sam and Everingham, Mark},
  booktitle={bmvc},
  volume={2},
  number={4},
  pages={5},
  year={2010},
  organization={Citeseer}
}
```
</details>

下载具有高分辨率的LSP数据集[LSP dataset original](http://sam.johnson.io/research/lsp_dataset_original.zip)，解压至文件夹`$MMHUMAN3D/data/datasets`，并使其具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── lsp
            ├── images
            |  ├── im0001.jpg
            |  ├── im0002.jpg
            |  └── ...
            └── joints.mat
```

### LSPET

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://sam.johnson.io/research/publications/johnson11cvpr.pdf">LSP-Extended (CVPR'2011)</a></summary>

```bibtex
@inproceedings{johnson2011learning,
  title={Learning effective human pose estimation from inaccurate annotation},
  author={Johnson, Sam and Everingham, Mark},
  booktitle={CVPR 2011},
  pages={1465--1472},
  year={2011},
  organization={IEEE}
}
```

</details>

下载具有高分辨率的LSPET数据集[HR-LSPET](http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip)，解压至文件夹`$MMHUMAN3D/data/datasets`，并使其具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── lspet
            ├── im00001.jpg
            ├── im00002.jpg
            ├── im00003.jpg
            ├── ...
            └── joints.mat
```



### MPI-INF-3DHP

<details>
<summary align="right"><a href="https://arxiv.org/pdf/1611.09813.pdf">MPI_INF_3DHP (3DV'2017)</a></summary>

```bibtex
@inproceedings{mono-3dhp2017,
 author = {Mehta, Dushyant and Rhodin, Helge and Casas, Dan and Fua, Pascal and Sotnychenko, Oleksandr and Xu, Weipeng and Theobalt, Christian},
 title = {Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision},
 booktitle = {3D Vision (3DV), 2017 Fifth International Conference on},
 url = {http://gvv.mpi-inf.mpg.de/3dhp_dataset},
 year = {2017},
 organization={IEEE},
 doi={10.1109/3dv.2017.00064},
}
```
</details>

修改[配置文件](https://github.com/open-mmlab/mmhuman3d/blob/main/tools/convert_datasets.py)中的`mpi_inf_3dhp`，从原始视频中提取图像。
请注意，这会花费较长的时间。

Config:
```python
mpi_inf_3dhp=dict(
  type='MpiInf3dhpConverter',
  modes=['train', 'test'],
  extract_img=True),  # 从原视频中提取图像，设置为True
```

下载[MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)数据集, 解压至文件夹`$MMHUMAN3D/data/datasets`, 使其具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── mpi_inf_3dhp
            ├── mpi_inf_3dhp_test_set
            │   ├── TS1
            │   ├── TS2
            │   ├── TS3
            │   ├── TS4
            │   ├── TS5
            │   └── TS6
            ├── S1
            │   ├── Seq1
            │   └── Seq2
            ├── S2
            │   ├── Seq1
            │   └── Seq2
            ├── S3
            │   ├── Seq1
            │   └── Seq2
            ├── S4
            │   ├── Seq1
            │   └── Seq2
            ├── S5
            │   ├── Seq1
            │   └── Seq2
            ├── S6
            │   ├── Seq1
            │   └── Seq2
            ├── S7
            │   ├── Seq1
            │   └── Seq2
            └── S8
                ├── Seq1
                └── Seq2
```



### MPII

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

下载[MPII](http://human-pose.mpi-inf.mpg.de/#download)数据集，并从[这里](https://github.com/princeton-vl/pose-hg-train/tree/master/data/mpii/annot?rgh-link-date=2020-07-05T04%3A14%3A02Z)下载标注。
解压至文件夹`$MMHUMAN3D/data/datasets`, 使其具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── mpii
            |── train.h5
            └── images
                |── 000001163.jpg
                |── 000003072.jpg
                └── ...
```


### PoseTrack18

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Andriluka_PoseTrack_A_Benchmark_CVPR_2018_paper.html">PoseTrack18 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{andriluka2018posetrack,
  title={Posetrack: A benchmark for human pose estimation and tracking},
  author={Andriluka, Mykhaylo and Iqbal, Umar and Insafutdinov, Eldar and Pishchulin, Leonid and Milan, Anton and Gall, Juergen and Schiele, Bernt},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5167--5176},
  year={2018}
}
```

</details>

下载[PoseTrack18](https://posetrack.net/users/download.php)数据集，解压至文件夹`$MMHUMAN3D/data/datasets`，使其具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── posetrack
            ├── images
            │   ├── train
            │   │   ├── 000001_bonn_train
            │   │   │   ├── 000000.jpg
            │   │   │   ├── 000001.jpg
            │   │   │   └── ...
            │   │   └── ...
            │   ├── val
            │   │   ├── 000342_mpii_test
            │   │   │   ├── 000000.jpg
            │   │   │   ├── 000001.jpg
            │   │   │   └── ...
            │   │   └── ...
            │   └── test
            │       ├── 000001_mpiinew_test
            │       │   ├── 000000.jpg
            │       │   ├── 000001.jpg
            │       │   └── ...
            │       └── ...
            └── posetrack_data
                └── annotations
                    ├── train
                    │   ├── 000001_bonn_train.json
                    │   ├── 000002_bonn_train.json
                    │   └── ...
                    ├── val
                    │   ├── 000342_mpii_test.json
                    │   ├── 000522_mpii_test.json
                    │   └── ...
                    └── test
                        ├── 000001_mpiinew_test.json
                        ├── 000002_mpiinew_test.json
                        └── ...
```

### Penn Action

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_iccv_2013/papers/Zhang_From_Actemes_to_2013_ICCV_paper.pdf">Penn Action (ICCV'2013)</a></summary>

```bibtex
@inproceedings{zhang2013pennaction,
 title={From Actemes to Action: A Strongly-supervised Representation for Detailed Action Understanding},
 author={Zhang, Weiyu and Zhu, Menglong and Derpanis, Konstantinos},
 booktitle={ICCV},
 year={2013}
}
```

</details>

下载[Penn Action](https://upenn.box.com/PennAction)数据集，解压至文件夹`$MMHUMAN3D/data/datasets`，使其具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
   └── datasets
       └── penn_action
           ├── frames
           │   ├── 0001
           │   │   ├── 000001.jpg
           │   │   ├── 000002.jpg
           │   │   └── ...
           │   └── ...
           └── labels
               ├── 0001.mat
               ├── 0002.mat
               └── ...
```

### PW3D

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Timo_von_Marcard_Recovering_Accurate_3D_ECCV_2018_paper.pdf">PW3D (ECCV'2018)</a></summary>

```bibtex
@inproceedings{vonMarcard2018,
title = {Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera},
author = {von Marcard, Timo and Henschel, Roberto and Black, Michael and Rosenhahn, Bodo and Pons-Moll, Gerard},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2018},
month = {sep}
}
```

</details>

下载[PW3D](https://virtualhumans.mpi-inf.mpg.de/3DPW/)数据集，解压至文件夹`$MMHUMAN3D/data/datasets`，并使其具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── pw3d
            |── imageFiles
            |   |    └── courtyard_arguing_00
            |   |       ├── image_00000.jpg
            |   |       ├── image_00001.jpg
            |   |       └── ...
            └── sequenceFiles
                ├── train
                │   ├── downtown_arguing_00.pkl
                │   └── ...
                ├── val
                │   ├── courtyard_arguing_00.pkl
                │   └── ...
                └── test
                    ├── courtyard_basketball_00.pkl
                    └── ...

```



### SPIN

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/pdf/1909.12828.pdf">SPIN (ICCV'2019)</a></summary>

```bibtex
@inproceedings{kolotouros2019spin,
  author = {Kolotouros, Nikos and Pavlakos, Georgios and Black, Michael J and Daniilidis, Kostas},
  title = {Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop},
  booktitle={ICCV},
  year={2019}
}
```

</details>

下载经过处理的[.npz 文件](https://github.com/nkolot/SPIN/blob/master/fetch_data.sh)，使文件夹具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── datasets
        └── spin_data
            ├── coco_2014_train.npz
            ├── hr-lspet_train.npz
            ├── lsp_dataset_original_train.npz
            ├── mpi_inf_3dhp_train.npz
            └── mpii_train.npz
```



### SURREAL

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/pdf/1701.01370.pdf">SURREAL (CVPR'2017)</a></summary>

```bibtex
@inproceedings{varol17_surreal,
 title     = {Learning from Synthetic Humans},
 author    = {Varol, G{\"u}l and Romero, Javier and Martin, Xavier and Mahmood, Naureen and Black, Michael J. and Laptev, Ivan and Schmid, Cordelia},
 booktitle = {CVPR},
 year      = {2017}
}
```

</details>

下载[SURREAL](https://www.di.ens.fr/willow/research/surreal/data/)数据集，并使其具有如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
   └── datasets
       └── surreal
          ├── train
          │   ├── run0
          |   |    ├── 03_01
          |   |    │   ├── 03_01_c0001_depth.mat
          |   |    │   ├── 03_01_c0001_info.mat
          |   |    │   ├── 03_01_c0001_segm.mat
          |   |    │   ├── 03_01_c0001.mp4
          |   |    │   └── ...
          |   |    └── ...
          │   ├── run1
          │   └── run2
          ├── val
          │   ├── run0
          │   ├── run1
          │   └── run2
          └── test
              ├── run0
              ├── run1
              └── run2
```


### VIBE

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/pdf/1912.05656.pdf">VIBE (CVPR'2020)</a></summary>

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

</details>

请下载经过处理的`mpi_inf_3dhp`与`pw3d`的[.npz 文件](https://github.com/nkolot/SPIN/blob/master/fetch_data.sh)，以及预训练之后的特征提取权重 [spin.pth](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/vibe/spin.pth?versionId=CAEQHhiBgIDrxqbU6xciIGIzOWFkMWYyNzMwMjRhMzBiYzM3NDFiMmVkY2JkZTVh)。 将文件夹整理成如下的结构:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    ├── checkpoints
    |   └── spin.pth
    └── datasets
        └── vibe_data
            ├── mpi_inf_3dhp_train.npz
            └── pw3d_test.npz
```
