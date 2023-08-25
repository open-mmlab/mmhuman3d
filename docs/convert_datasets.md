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
    
2. For using HumanData only (not including converting), you just need to do several changes to the file structues on the originally downloaded datasets. The converting process and file included will be marked **"Converter"**.
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
      --datasets <dataset name/perfix> \ 
      --root_path <data root dir> \
      --output_path <output dir> \
      --modes train
     # dataset folder should be os.path.join(root_path, datasets)
     # output path is the folder which sotres output HumanData
     ```
7. (For convert only) For discussing processing speed or time, hardware config is RTX3090 + i9-12900 + 64GB RAM + Cuda11.8 & Python3.9. The most resource-consuming processes are calling SMPL/SMPLX models and camera projection.   

## Overview - Current Supported Datasets
- EgoBody
- GTA-Human++
- H36M (Neural Annot)
- MPII (Neural Annot)
- MSCOCO (Neural Annot)
- PW3D (Neural Annot)

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

This process converts the coco annotation json to faciliate sorting ids.
```
python tools/preprocess/neural_annot.py --dataset_path /YOUR_PATH/mpii
```

**Step 3 (Converter) - Convert Dataset**

Process speed is ~60item/s, totally ~17k item, and takes several minutes.
```
python tools/convert_datasets.py \
 --datasets mpii \
 --root_path /mnt/d/datasets \
 --output_path /mnt/d/datasets/mpii/output \
 --modes train
```
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

This process converts the coco annotation json to faciliate sorting ids.
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

This process converts the coco annotation json to faciliate sorting ids.
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

## Subsection 2 - Synthetic Datasets

<details>
<summary>AGORA (IP)</summary>

</details>


<details>
<summary>GTA-Human++ (IP)</summary>

</details>

## Subsection 3 - Real Multi-Human Datasets
<details>
<summary>EgoBody</summary>

**Step 1 - Only Step for using HumanData**

For Egobody dataset, please download as instructed in [HomePage](https://github.com/sanweiliti/EgoBody).
Egobody dataset can be split as two distinct subset, "egocentric" which denotes the views from human-wear kinect (single view, single person) and "kinect" which denotes the the fixed various kinect (multi-view, two person).

**Step 2 (Converter) - Convert Dataset**

There are 6 HumanData set, seperated by "train, test and val", and "egocentric and kinect". For example, "egocentric_train" & "kinect_train" comprise the train set as mentioned in the paper. 

```
available_modes=['egocentric_train', 'egocentric_test', 'egocentric_val',
                 'kinect_train', 'kinect_test', 'kinect_val']
```

Kinect set conducts sequence-level processing whereas egocentric set only processes image-level and is much slower (due to changing camera parameters). Total processing time should be **1-3 days**. 

```
python tools/convert_datasets.py \
    --datasets egobody \
    --root_path /mnt/d/datasets \
    --output_path /mnt/d/datasets/egobody/output \
    #  --modes egocentric_train
```
</details>

<details>
<summary>Ubody (IP)</summary>

</details>



## Subsection 4 - Real Single-Human Datasets
<details>
<summary>SSP3D</summary>
   
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
</details>

