## Data preparation
For training and testing, datasets used are listed as following:
1. [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
2. [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
3. [MPII](http://human-pose.mpi-inf.mpg.de)
4. [COCO](http://cocodataset.org/#home)
5. [MPI_INF_3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)
Please go to the official website to download them on your computer.

More specifically:
The overall data folder structure should be like this:
```
${ROOT_PATH}
|--h36m
|   |--S1
|   |   |--Videos
|   |   |--Segments
|   |   |--Bboxes
|   |--S5
|   |   |--Videos
|   |   |--Segments
|   |   |--Bboxes
|   ...
|   |--S11
|       |--Videos
|       |--Segments
|       |--Bboxes
|--3DPW
|   |--imageFiles
|   |--sequenceFiles
|--mpii
|   |--images
|   |--train.h5
|--coco
|   |--annotations
|   |--train2014
|--mpi_inf_3dhp
    |--mpi_inf_3dhp_test_set
    |--S1
    |--S2
    ...
    |--S8

```

### Generate dataset files
After preparing the data, then simply run this command:
```
python tools/preprocess.py \
--datasets all \
--root_path $YOUR_ROOT_PATH \
--output_path $YOUR_OUTPUT_PATH
```
After this command finished, you can see the preprocessed npz files in the YOUR_OUTPUT_PATH.
