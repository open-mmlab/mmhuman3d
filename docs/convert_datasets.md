# Convert Datasets to HumanData and Preparation Before Training
**This page describes how to use the various HumanData flies** as well as (only if you are interested) how to process various datasets to current HumanData.

## Introduction & Settings
1. How can I read image from HumanData image path?

   In HumanData file, image path denotes the relative path within the dataset. In this example:

    ```
    your_path_to_dataset = '/mnt/d'
    dataset_name = 'egobody'
    image_path = HumanData['image_path'][100]  # for example
    image_real_path = os.path.join(your_path_to_dataset, dataset_name, image_path)
    image = cv2.imread(image_real_path)
    ```

2. For using HumanData only (not including converting), you just need to do several changes to the file structures on the originally downloaded datasets. The converting process and file included will be marked **"Converter"**.
3. Ideal size of HumanData file is less than 2GB, given loading it uses 10x memory, therefore some huge datasets are split in several files.
4. All the middle-stage pre-process files are **not required** when using HumanData annotations.
5. (For convert only) Install "convertors" branch and set the working directory as below.

     ```
     git clone -b convertors https://github.com/open-mmlab/mmhuman3d.git
     cd /Your_path_to_mmhuman3d/mmhuman3d  # e.g. /mnt/d/zoehuman/mmhuman3d
     ```
6. (For convert only) The usage of convert_datasets.py (Example), if "--modes" is not specified, all avalilable modes (all data) will be processed.
     ```
     python tools/convert_datasets.py \
      --datasets <dataset name/prefix> \
      --root_path <data root dir> \
      --output_path <output dir> \
      --modes train
     # dataset folder should be os.path.join(root_path, datasets)
     # output path is the folder which sotres output HumanData
     ```
7. (For convert only) For discussing processing speed or time, hardware config is RTX3090 + i9-12900 + 64GB RAM + Cuda11.8 & Python3.9. The most resource-consuming processes are calling SMPL/SMPLX models and camera projection.  

## Overview - Current Supported Datasets
- Behave
- CHI3D
- EgoBody
- FIT3D
- EMDB
- GTA-Human++
- H36M (Neural Annot)
- HumanSC3D
- MPII (Neural Annot)
- MSCOCO (Neural Annot)
- PW3D (Neural Annot)
- Slpoer4D

## Subsection 1 - Neural Annot Datasets
Overall: Download from [Nerual Annot Homepage](https://github.com/mks0601/NeuralAnnot_RELEASE/blob/main/README.md)

<details>
<summary>H36M</summary>

**Step 1 - Only Step for using HumanData**

Download the original data and SMPLX annotation and rearrange the file structure as below:

```
D:\datasets\h36m\
│
├── annotations\
│   ├──Human36M_subject11_SMPLX_NeuralAnnot.json
│   ├──Human36M_subject11_camera.json
│   ├──Human36M_subject11_data.json
│   ├──Human36M_subject11_joint_3d.json
│
├── images\
│   ├── s_01_act_02_subact_01_ca_01\
│   ├── s_01_act_02_subact_01_ca_02\
```

**Step 2 (Converter) - Convert Dataset**

Due to size considerations, "train" and "val" set is separated in to 5 and 2 files (depends on subject id). HumanData file ranges from 900MB to 1.5GB. Processing speed is ~40item/s, total time is about **12 hours**.

```
python tools/convert_datasets.py \
 --datasets h36m\
 --root_path /mnt/d/datasets \
 --output_path /mnt/d/datasets/h36m/output \
 --modes train
```
</details>

<details>
<summary>MPII</summary>

**Step 1 - Only Step for using HumanData**

Download and rearrange the file structure as below:

```
E:\mpii\
│
├── annotations\
│   ├──MPII_train_SMPLX_NeuralAnnot.json
│   ├──test.json
│   ├──train.json
│   └──train_reformat.json  # Not required in HumanData
│
├── images\
```
**Step 2 (Converter) - Preprocess coco annotations**

This process converts the coco annotation json to facilitate sorting ids.
```
python tools/preprocess/neural_annot.py --dataset_path /YOUR_PATH/mpii
```

**Step 3 (Converter) - Convert Dataset**

Process speed is ~60item/s, totally ~17k item, and takes several minutes.

</details>

<details>
<summary>MSCOCO</summary>

**Step 1 - Only Step for using HumanData**

Download and rearrange the file structure as below:

```
D:\datasets\mscoco\
│
├── annotations\
│   ├──MSCOCO_train_SMPLX.json
│   ├──MSCOCO_train_SMPLX_all_NeuralAnnot.json
│   ├──coco_wholebody_train_v1.0.json
│   ├──coco_wholebody_train_v1.0_reformat.json  # Not required in HumanData, generated in Step 2 and used in step 3
│   └──coco_wholebody_val_v1.0.json
│
├── images\
│   ├── train2017\
│   └── val2017\
```
**Step 2 (Converter) - Preprocess coco annotations**

This process converts the coco annotation json to facilitate sorting ids.
```
python tools/preprocess/neural_annot.py --dataset_path /YOUR_PATH/mscoco
```

**Step 3 (Converter) - Convert Dataset**/

Processing speed is ~50item/s, total ~260k item.
```
python tools/convert_datasets.py \
 --datasets mscoco \
 --root_path /mnt/d/datasets \
 --output_path /mnt/d/datasets/mscoco/output \
 --modes train
```
</details>

<details>
<summary>PW3D</summary>

**Step 1 - Only Step for using HumanData**

Download and rearrange the file structure as below:

Note: Rename "*validation*" as "*val*"
```
D:\datasets\pw3d\
│
├── imageFiles\
│   ├── courtyard_arguing_00\
│   ├── courtyard_backpack_00\
│
├──3DPW_test.json
├──3DPW_test_SMPLX_NeuralAnnot.json
├──3DPW_train.json
├──3DPW_train_SMPLX_NeuralAnnot.json
├──3DPW_val.json
└──3DPW_val_SMPLX_NeuralAnnot.json
```
**Step 2 (Converter) - Preprocess coco annotations**

This process converts the coco annotation json to facilitate sorting ids.
```
python tools/preprocess/neural_annot.py --dataset_path /YOUR_PATH/pw3d
```

**Step 3 (Converter) - Convert Dataset**

3DPW datasets contains 3 modes ("train", "test" and "val"), processing speed is ~40item/s, total 60k item.
```
python tools/convert_datasets.py \
 --datasets pw3d \
 --root_path /mnt/d/datasets \
 --output_path /mnt/d/datasets/pw3d/output \
 --modes train test val
```
</details>

<details>
<summary>Reference of Neural Annot</summary>

```
@InProceedings{Moon_2022_CVPRW_NeuralAnnot,  
author = {Moon, Gyeongsik and Choi, Hongsuk and Lee, Kyoung Mu},  
title = {NeuralAnnot: Neural Annotator for 3D Human Mesh Training Sets},  
booktitle = {Computer Vision and Pattern Recognition Workshop (CVPRW)},  
year = {2022}  
}  

@InProceedings{Moon_2023_CVPRW_3Dpseudpgts,  
author = {Moon, Gyeongsik and Choi, Hongsuk and Chun, Sanghyuk and Lee, Jiyoung and Yun, Sangdoo},  
title = {Three Recipes for Better 3D Pseudo-GTs of 3D Human Mesh Estimation in the Wild},  
booktitle = {Computer Vision and Pattern Recognition Workshop (CVPRW)},  
year = {2023}  
}  
```
</details>

## Subsection 2 - Synthetic Datasets

<details>
<summary>AGORA (IP)</summary>

</details>


<details>
<summary>GTA-Human++ (IP)</summary>

</details>

<details>
<summary>SPEC (IP)</summary>

</details>

## SPEC (Code not added yet)



<details>
<summary>SynBody (IP)</summary>

</details>

## Subsection 3 - Real Multi-Human Datasets
<details>
<summary>EgoBody</summary>

## EgoBody: Human Body Shape and Motion of Interacting People from Head-Mounted Devices

**Dataset Split**

There are 6 HumanData set, separated by "train, test and val", and "egocentric and kinect". For example, "egocentric_train" & "kinect_train" comprise the train set as mentioned in the paper.

```
available_modes=['egocentric_train', 'egocentric_test', 'egocentric_val',
                 'kinect_train', 'kinect_test', 'kinect_val']
# HumanData Name is "egobody_<mode>_<some other index>.npz"
```

**Step 1 - Only Step for using HumanData**

For Egobody dataset, please download as instructed in [HomePage](https://github.com/sanweiliti/EgoBody).
Egobody dataset can be split as two distinct subset, "egocentric" which denotes the views from human-wear kinect (single view, single person) and "kinect" which denotes the the fixed various kinect (multi-view, two person).

**Step 2 (Converter) - Convert Dataset**

Kinect set conducts sequence-level processing whereas egocentric set only processes image-level and is much slower (due to changing camera parameters). Total processing time should be **1-3 days**.

```
python tools/convert_datasets.py \
    --datasets egobody \
    --root_path /mnt/d/datasets \
    --output_path /mnt/d/datasets/egobody/output \
    #  --modes egocentric_train
```

**Citations**
```
@inproceedings{Zhang:ECCV:2022,
   title = {EgoBody: Human Body Shape and Motion of Interacting People from Head-Mounted Devices},
   author = {Zhang, Siwei and Ma, Qianli and Zhang, Yan and Qian, Zhiyin and Kwon, Taein and Pollefeys, Marc and Bogo, Federica and Tang, Siyu},
   booktitle = {European conference on computer vision (ECCV)},
   month = oct,
   year = {2022}
}
```

</details>

<details>
<summary>Ubody</summary>


## Ubody

**Dataset Split**

Ubody dataset splits the train/test sets from two protocols as follow:
```
Intra-scene: in each scene, the former 70% of the
    videos are the training set, and the last 30% are the test set.
Inter-scene: we use ten scenes of the videos as the
    training set and the other five scenes as the test set.
```
Therefore, there are 4 HumanData files, with name "intra_scene_train", "intra_scene_test", "inter_scene_train" and "inter_scene_test".
Note that "train" & "test" from each protocol comprises the whole dataset, please do not mix use.


**Step 1 - Split Videos and arrange files**

For Ubody dataset, please request data and download as instructed in [GitHub HomePage](https://github.com/IDEA-Research/OSX).

Git clone the repo and run code to split videos, this process usually takes **12hours-1day**.

```
python video2image.py --video_folder <your path to ubody dataset>
```
After that, the file structure should be like this:
```
|-- UBody
|   |-- images
|   |-- videos
|   |-- annotations
|   |-- splits
|   |   |-- inter_scene_test_list.npy
|   |   |-- intra_scene_test_list.npy
```

**Step 2 (Converter) - Convert Dataset**

Note there will be some middle files generated in the root_path, which can be deleted after conversion. Total process time should be about **1day**.

```
python tools/convert_datasets.py \
    --datasets ubody \
    --root_path /mnt/d/datasets \
    --output_path /mnt/d/datasets/ubody/output \
    --modes inter intra
```

**Citations**
```
@article{lin2023one,
  title={One-Stage 3D Whole-Body Mesh Recovery with Component Aware Transformer},
  author={Lin, Jing and Zeng, Ailing and Wang, Haoqian and Zhang, Lei and Li, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```

</details>

## Subsection 4 - Real Single-Human Datasets

<details>
<summary>Behave</summary>

## BEHAVE dataset

**Data Split**

We follow the same split ("train" and "test") as in Paper.

**Step 1 - Only Step for using HumanData**

Download as instructed in [Github HomePage](https://github.com/xiexh20/behave-dataset). the extracted dataset should be like this:
```
E:\behave\
├── behave-30fps-params-v1\
│   ├── Date01_Sub01_backpack_back\
│   │   ├──info.json
│   │   ├──object_fit_all.npz
│   │   └──smpl_fit_all.npz
│   ├── Date01_Sub01_backpack_hand\
├── calibs\
│   ├── Date01\
│   │   ├── background\
│   │   │   ├── t0002.000\
│   │   │   └──background.ply
│   │   └── config\
│   │       ├── 0\
│   │       ├── 1\
│   │       ├── 2\
│   │       └── 3\
│   ├── Date02\
├── objects\
│   ├── backpack\
│   │   ├──backpack.obj
│   │   ├──backpack.obj.mtl
│   │   ├──backpack.png
│   │   ├──backpack_f1000.ply
│   ├── basketball\
├── sequences\
│   ├── Date01_Sub01_backpack_back\
│   │   ├── t0005.000\
│   │   │   ├── backpack\
│   │   │   ├── person\
│   │   │   ├──k0.color.json
│   │   ├── t0006.000\
│   ├── Date01_Sub01_backpack_hand\
└──split.json
```

**Step 2 (Converter) - Convert Datasets**

The data processing speed is ~2item/s, total take taken would be **2~3hours**.
```
python tools/convert_datasets.py  \
    --datasets behave  \
    --root_path /mnt/d/datasets \
    --output_path /mnt/d/datasets/behave/output \
    --modes train test
```

**Citations**
```bibtex
@inproceedings{bhatnagar22behave,
  title={Behave: Dataset and method for tracking human object interactions},
  author={Bhatnagar, Bharat Lal and Xie, Xianghui and Petrov, Ilya A and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15935--15946},
  year={2022}
}
```

</details>


<details>
<summary>EHF</summary>

## EHF Dataset

**Step 1 - Only Step for using HumanData**

Download dataset from [SMPLX HomePage](https://smpl-x.is.tue.mpg.de/), the extracted dataset should be like this:
```
E:\ehf\
├──01_2Djnt.json
├──01_2Djnt.png
├──01_align.ply
├──01_img.png
├──01_scan.obj
├──02_2Djnt.json
├──02_2Djnt.png
├──02_align.ply
├──02_img.png
├──02_scan.obj
```

**Step 2 (Converter) - Preprocess: Fit SMPLX Parameters from Mesh**

In Ehf datasets, data of smplx instances are provided in "xx_align.ply" mesh format. This step fits the smplx parameters from the mesh and save the parameters in "xx_align.npz" format, which is an optimization process and takes ~20sec/instance. Final error MSE is ~0.1mm.

```
python tools/preprocess/fit_shape2smplx.py --load_dir <ehf dataset path> --mesh_type ply
```

**Step 3 (Converter) - Convert Datasets**

The convert process should finish in several seconds.
```
python tools/convert_datasets.py \
    --datasets ehf \
    --root_path /mnt/e \
    --output_path /mnt/e/ehf/output \
    --modes val
```

**Citations**
```
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: {3D} Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10975--10985},
  year = {2019}
}
```


</details>


<details>
<summary>EMDB</summary>

## EMDB: The Electromagnetic Database of Global 3D Human Pose and Shape in the Wild

**Data Split**

This dataset is not specified in paper, so all data is used for training.

**Step 1 - Only Step for using HumanData**

Request and download dataset from [HomePage](https://eth-ait.github.io/emdb/).
Extarct ZIP files and the extracted dataset should be like this:
```
D:\datasets\emdb\
├── P0\
│   ├── 00_mvs_a\
│   ├── 01_mvs_b\
│   ├── 02_mvs_c\
│   ├── 03_mvs_d\
│   ├── 04_mvs_e\
├── P1\
├── P2\
├── P3\
├── P4\
├── P5\
├── P6\
├── P7\
├── P8\
├── P9\
```

**Step 2 (Converter) - Convert Datasets**

The convert process should finish in **<1hour**.

```
python tools/convert_datasets.py \
    --datasets emdb \
    --root_path /mnt/d/datasets \
    --output_path /mnt/d/datasets/emdb/output \
    --modes train
```

**Citations**
```bibtex
@inproceedings{kaufmann2023emdb,
  author = {Kaufmann, Manuel and Song, Jie and Guo, Chen and Shen, Kaiyue and Jiang, Tianjian and Tang, Chengcheng and Z{\'a}rate, Juan Jos{\'e} and Hilliges, Otmar},
  title = {{EMDB}: The {E}lectromagnetic {D}atabase of {G}lobal 3{D} {H}uman {P}ose and {S}hape in the {W}ild},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year = {2023}
}
```

</details>





<details>
<summary>SSP3D</summary>

## SSP3D

**Step 1 - Only Step for using HumanData**

Download dataset from [HomePage](https://github.com/akashsengupta1997/SSP-3D)

**Step 2 (Converter) - Convert Dataset**

This is a very small dataset which should finish in a gilmpse.
```
python tools/convert_datasets.py \
    --datasets ssp3d \
    --root_path /mnt/e \
    --output_path /mnt/e/ssp3d/output
```

**Citations**
```
@InProceedings{STRAPS2018BMVC,
               author = {Sengupta, Akash and Budvytis, Ignas and Cipolla, Roberto},
               title = {Synthetic Training for Accurate 3D Human Pose and Shape Estimation in the Wild},
               booktitle = {British Machine Vision Conference (BMVC)},
               month = {September},
               year = {2020}  
}
```

</details>
