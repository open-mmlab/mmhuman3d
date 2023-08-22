# Convert Datasets to HumanData and Use in Training
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
6. (For convert only) The usage of convert_datasets.py (Example)
     ```
     python tools/convert_datasets.py \
      --datasets <dataset name/perfix> \ 
      --root_path <data root dir> \
      --output_path <output dir> \
      --modes train
     # dataset folder should be os.path.join(root_path, datasets)
     # output path is the folder which sotres output HumanData
     ``` 

## Overview - Current Supported Datasets
- MSCOCO (Neural Annot)

## Subsection 1 - Neural Annot Datasets
Overall: Download from [Nerual Annot Homepage](https://github.com/mks0601/NeuralAnnot_RELEASE/blob/main/README.md)
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
│   │
│   ├── train2017\
│   │
│   └── val2017\
```
**Step 2 (Converter) - Preprocess coco annotations**

This process converts the coco annotation json to faciliate sorting ids.
```
python tools/preprocess/neural_annot.py --dataset_path /YOUR_PATH/mscoco
```

**Step 3 (Converter) - Convert Dataset**
```
python tools/convert_datasets.py \
 --datasets mscoco \
 --root_path /mnt/d/datasets \
 --output_path /mnt/d/datasets/mscoco/output \
 --modes train
```
</details>

<details>
<summary>MPII</summary>
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
```
python tools/convert_datasets.py \
 --datasets pw3d \
 --root_path /mnt/d/datasets \
 --output_path /mnt/d/datasets/pw3d/output \
 --modes train test val
```
</details>
