## Data preparation


### Overview

Our data pipeline use [HumanData](mmhuman3d/docs/human_data.md) structure for
storing and loading. The proprocessed npz files can be obtained from raw data using our data converters, and the supported configs can be found [here](mmhuman3d/tools/convert_datasets.py).


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

### Datasets for supported algorithms

For HMR training and testing, the following datasets are required:
  - [COCO](#coco)
  - [Human3.6M](#human36m)
  - [Human3.6M Mosh](#human36mmosh)
  - [MPI-INF-3DHP](#mpi-inf-3dhp)
  - [MPII](#mpii)
  - [LSP](#lsp)
  - [LSPET](#lspet)
  - [PW3D](#pw3d)

```
dataset-name: coco, pw3d, mpii, mpi_inf_3dhp, lsp_original, lsp_extended, h36m
```

For SPIN training, the following datasets are required:
  - [COCO](#coco)
  - [Human3.6M](#human36m)
  - [Human3.6M Mosh](#human36mmosh)
  - [MPI-INF-3DHP](#mpi-inf-3dhp)
  - [MPII](#mpii)
  - [LSP](#lsp)
  - [LSPET](#lspet)
  - [PW3D](#pw3d)
  - [SPIN](#spin)

```
dataset-name: spin, h36m_spin
```

For HYBRIK training and testing, the following datasets are required:
  - [HybrIK](#hybrik)
  - [COCO](#coco)
  - [Human3.6M](#human36m)
  - [MPI-INF-3DHP](#mpi-inf-3dhp)
  - [PW3D](#pw3d)

```
dataset-name: h36m_hybrik, pw3d_hybrik, mpi_inf_3dhp_hybrik, coco_hybrik
```

## COCO

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

For [COCO](http://cocodataset.org/) data, please download from [COCO download](http://cocodataset.org/#download). COCO'2014 Train is needed for HMR training and COCO'2017 Train is needed for HybrIK trainig.
Download and extract them under  `$MMHUMAN3D/data/datasets`, and make them look like this:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        │── annotations
            ├── person_keypoints_train2014.json
            ├── person_keypoints_val2014.json
        │── coco
            │-- train2014
            │   ├── COCO_train2014_000000000009.jpg
            │   ├── COCO_train2014_000000000025.jpg
            │   ├── COCO_train2014_000000000030.jpg
            |   │-- ...
            │-- train_2017
                │── annotations
                │   ├── person_keypoints_train2017.json
                │   ├── person_keypoints_val2017.json
                │── train2017
                │   ├── 000000000009.jpg
                │   ├── 000000000025.jpg
                │   ├── 000000000030.jpg
                │   │-- ...
                │── val2017
                    ├── 000000000139.jpg
                    ├── 000000000285.jpg
                    ├── 000000000632.jpg
                    │-- ...
```


## COCO-WholeBody

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

For [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody/) datatset, images can be downloaded from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
Download and extract them under `$MMHUMAN3D/data/datasets`, and make them look like this:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        │── annotations
        │   │-- coco_wholebody_train_v1.0.json
        │   |-- coco_wholebody_val_v1.0.json
        │── coco
            │-- train_2017
                │── annotations
                │   ├── person_keypoints_train2017.json
                │   ├── person_keypoints_val2017.json
                │── train2017
                │   ├── 000000000009.jpg
                │   ├── 000000000025.jpg
                │   ├── 000000000030.jpg
                │   │-- ...
                │── val2017
                    ├── 000000000139.jpg
                    ├── 000000000285.jpg
                    ├── 000000000632.jpg
                    │-- ...

```




## CrowdPose

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

For [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) data, please download from [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose).
Download and extract them under `$MMHUMAN3D/data/datasets`, and make them look like this:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        │── Crowdpose
            │-- crowdpose_train.json
            │-- crowdpose_val.json
            │-- crowdpose_trainval.json
            │-- crowdpose_test.json
            │-- images
                │-- 100000.jpg
                │-- 100001.jpg
                │-- 100002.jpg
                │-- ...
```


## Human3.6M

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

For [Human3.6M](http://vision.imar.ro/human3.6m/description.php), please download from the official website and run the [preprocessing script](mmhuman3d/data/data_converters/h36m.py), which will extract pose annotations at downsampled framerate (10 FPS). The processed data should have the following structure:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        ├── h36m
            ├── annot
            ├── S1
            |   ├── images
            |   |    ├── S1_Directions_1.54138969
            |   |       ├── S1_Directions_1.54138969_00001.jpg
            |   |       ├── S1_Directions_1.54138969_00002.jpg
            |   |       ├── ...
            |   ├── ...
            |   ├── MyPoseFeatures
            |   ├── MySegmentsMat
            |   ├── Videos
            ├── S5
            ├── S6
            ├── S7
            ├── S8
            ├── S9
            ├── S11
            `── metadata.xml
```

## Human3.6M Mosh

<!-- [DATASET] -->

For data preparation of [Human3.6M](http://vision.imar.ro/human3.6m/description.php) for HMR and SPIN training, we use the [MoShed](https://mosh.is.tue.mpg.de/) data provided in [HMR](https://github.com/akanazawa/hmr) for training. However, due to license limitations, we are not allowed to redistribute the data. Even if you do not have access to these parameters, you can still generate the preprocessed h36m npz file without mosh parameters using our [converter](mmhuman3d/data/data_converters/keypoints_mapping/h36m.py).

Config without mosh:
```python
h36m_p1=dict(
    type='H36mConverter',
    modes=['train', 'valid'],
    protocol=1,
    prefix='h36m'),
```

Config:
```python
h36m_p1=dict(
    type='H36mConverter',
    modes=['train', 'valid'],
    protocol=1,
    mosh_dir='data/datasets/h36m_mosh', # supply the directory to the mosh if available
    prefix='h36m'),
```

If you have MoShed data available, it should have the following structure:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        ├── h36m_mosh
            ├── annot
            ├── S1
            |   ├── images
            |   |    ├── Directions 1_cam0_aligned.pkl
            |   |    ├── Directions 1_cam1_aligned.pkl
            |   |    ├── ...
            ├── S5
            ├── S6
            ├── S7
            ├── S8
            ├── S9
            `── S11
```

## HybrIK

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

For [HybrIK](https://github.com/Jeff-sjtu/HybrIK), please download the parsed [json annotation files](https://github.com/Jeff-sjtu/HybrIK#fetch-data) and place them in the folder structure below:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        ├── hybrik_data
            ├── Sample_5_train_Human36M_smpl_leaf_twist_protocol_2.json
            ├── Sample_20_test_Human36M_smpl_protocol_2.json
            ├── 3DPW_test_new.json
            ├── annotation_mpi_inf_3dhp_train_v2.json
            `── annotation_mpi_inf_3dhp_test.json
```

To convert the preprocessed json files into npz files used for our pipeline,
run the following preprocessing scripts:
  - [Human3.6M](mmhuman3d/data/data_converters/h36m_hybrik.py)
  - [PW3D](mmhuman3d/data/data_converters/pw3d_hybrik.py)
  - [Mpi-Inf-3dhp](mmhuman3d/data/data_converters/mpi_inf_3dhp_hybrik.py)
  - [COCO](mmhuman3d/data/data_converters/coco_hybrik.py)


## LSP

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

For [LSP](https://sam.johnson.io/research/lsp.html), please download the high resolution version
[LSP dataset original](http://sam.johnson.io/research/lsp_dataset_original.zip).
Extract them under `$MMHUMAN3D/data/datasets`, and make them look like this:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        │── lsp_dataset_original
            ├── images
               ├── im0001.jpg
               ├── im0002.jpg
               └── ...
```

## LSPET

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

For [LSPET](https://sam.johnson.io/research/lspet.html), please download its high resolution form
[HR-LSPET](http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip).
Extract them under `$MMHUMAN3D/data/datasets`, and make them look like this:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        │── lspet_dataset
            ├── images
            │   ├── im00001.jpg
            │   ├── im00002.jpg
            │   ├── im00003.jpg
            │   └── ...
            └── joints.mat
```



## MPI-INF-3DHP

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

For [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/), download and extract them under `$MMHUMAN3D/data/datasets`, and make them look like this:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
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


## MPII

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

For [MPII](http://human-pose.mpi-inf.mpg.de/) data, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/).
Extract them under `$MMHUMAN3D/data/datasets`, and make them look like this:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        │── mpii
            |── train.h5
            `── images
                |── 000001163.jpg
                |── 000003072.jpg

```


## PoseTrack18

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

For [PoseTrack18](https://posetrack.net/users/download.php) data, please download from [PoseTrack18](https://posetrack.net/users/download.php).
Extract them under `$MMHUMAN3D/data/datasets`, and make them look like this:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        │── PoseTrack
            │-- images
            │   │-- train
            │   │   │-- 000001_bonn_train
            │   │   │   │-- 000000.jpg
            │   │   │   │-- 000001.jpg
            │   │   │   │-- ...
            │   │   │-- ...
            │   │-- val
            │   │   │-- 000342_mpii_test
            │   │   │   │-- 000000.jpg
            │   │   │   │-- 000001.jpg
            │   │   │   │-- ...
            │   │   │-- ...
            │   `-- test
            │       │-- 000001_mpiinew_test
            │       │   │-- 000000.jpg
            │       │   │-- 000001.jpg
            │       │   │-- ...
            │       │-- ...
            `-- posetrack_data
                │-- annotations
                    │-- train
                    │   │-- 000001_bonn_train.json
                    │   │-- 000002_bonn_train.json
                    │   │-- ...
                    │-- val
                    │   │-- 000342_mpii_test.json
                    │   │-- 000522_mpii_test.json
                    │   │-- ...
                    `-- test
                        │-- 000001_mpiinew_test.json
                        │-- 000002_mpiinew_test.json
                        │-- ...
```

## PW3D

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

For [PW3D](https://virtualhumans.mpi-inf.mpg.de/3DPW/) data, please download from [PW3D Dataset](https://virtualhumans.mpi-inf.mpg.de/3DPW/).
Extract them under `$MMHUMAN3D/data/datasets`, and make them look like this:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        │── 3DPW
            |── imageFiles
            |   |    ├── courtyard_arguing_00
            |   |       ├── image_00000.jpg
            |   |       ├── image_00001.jpg
            |   |       ├── ...
            `── sequenceFiles
                │-- train
                │   │-- downtown_arguing_00.pkl
                │   │-- ...
                │-- val
                │   │-- courtyard_arguing_00.pkl
                │   │-- ...
                `-- test
                    │-- courtyard_basketball_00.pkl
                    │-- ...

```



## SPIN

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

For [SPIN](https://github.com/nkolot/SPIN), please download the [preprocessed npz files](https://github.com/nkolot/SPIN/blob/master/fetch_data.sh) and place them in the folder structure below:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
`── data
    │── datasets
        ├── spin_data
            ├── coco_2014_train.npz
            ├── hr-lspet_train.npz
            ├── lsp_dataset_original_train.npz
            ├── mpi_inf_3dhp_train.npz
            `── mpii_train.npz
```
