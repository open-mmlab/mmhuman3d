# Getting Started

* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Body Model Preparation](#body-model-preparation)
* [Inference / Demo](#inference--demo)
  + [Single-person](#single-person)
  + [Multi-person](#multi-person)
* [Evaluation](#evaluation)
  + [Evaluate with a single GPU / multiple GPUs](#evaluate-with-a-single-gpu--multiple-gpus)
  + [Evaluate with slurm](#evaluate-with-slurm)
* [Training](#training)
  + [Training with a single / multiple GPUs](#training-with-a-single--multiple-gpus)
  + [Training with Slurm](#training-with-slurm)
* [More Tutorials](#more-tutorials)

## Installation

Please refer to [install.md](./install.md) for installation.

## Data Preparation

Please refer to [data_preparation.md](./preprocess_dataset.md) for data preparation.

## Body Model Preparation

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
  - Neutral model can be downloaded from [SMPLify](https://smplify.is.tue.mpg.de/).
  - All body models have to be renamed in `SMPL_{GENDER}.pkl` format. <br/>
    For example, `mv basicModel_neutral_lbs_10_207_0_v1.0.0.pkl SMPL_NEUTRAL.pkl`
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)

Download the above resources and arrange them in the following file structure:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── body_models
        ├── J_regressor_extra.npy
        ├── J_regressor_h36m.npy
        ├── smpl_mean_params.npz
        └── smpl
            ├── SMPL_FEMALE.pkl
            ├── SMPL_MALE.pkl
            └── SMPL_NEUTRAL.pkl
```

## Inference / Demo

```shell
python demo/estimate_smpl.py ${CONFIG_FILE} ${CHECKPOINT} [optional]
```

### Single-person
Optional arugments include:
- `--single_person_demo`: flag for single-person inference
- `--det_config`: MMDetection config
- `--det_checkpoint`: MMDetection checkpoint
- `--video_path`: input video load path
- `--output_path`: output video (with overlay) save path
- `--smooth_type`: smoothing mode

Example:
```shell
python demo/estimate_smpl.py \
    configs/hmr/resnet50_hmr_pw3d.py \
    data/checkpoints/resnet50_hmr_pw3d.pth \
    --single_person_demo \
    --det_config demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --video_path  demo/resources/single_person_demo.mp4 \
    --output_path vis_results/single_person_demo.mp4 \
    --smooth_type savgol
```

Note that the MMHuman3D checkpoints can be downloaded from the [model zoo](model_zoo.md).
Here we take HMR (resnet50_hmr_pw3d.pth) as an example.

### Multi-person
Optional arugments include:
- `--multi_person_demo`: flag for multi_person inference
- `--mmtracking_config`: MMTracking config
- `--video_path`: input video load path
- `--output_path`: output video (with overlay) save path
- `--smooth_type`: smoothing mode

Example 2: multi-person estimation
```shell
python demo/estimate_smpl.py \
    configs/hmr/resnet50_hmr_pw3d.py \
    data/checkpoints/resnet50_hmr_pw3d.pth \
    --multi_person_demo \
    --tracking_config demo/mmtracking_cfg/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py \
    --video_path  demo/resources/multi_person_demo.mp4 \
    --output_path vis_results/multi_person_demo.mp4 \
    --smooth_type savgol
```

## Evaluation

We provide pretrained models in the respective method folders in [config](https://github.com/open-mmlab/mmhuman3d/tree/main/configs).

### Evaluate with a single GPU / multiple GPUs

```shell
python tools/test.py ${CONFIG} --work-dir=${WORK_DIR} ${CHECKPOINT}
```
Example:
```shell
python tools/test.py configs/hmr/resnet50_hmr_pw3d.py --work-dir=work_dirs/hmr work_dirs/hmr/latest.pth
```

### Evaluate with slurm

If you can run MMHuman3D on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_test.sh`.

```shell
./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${WORK_DIR} ${CHECKPOINT}
```
Example:
```shell
./tools/slurm_test.sh my_partition test_hmr configs/hmr/resnet50_hmr_pw3d.py work_dirs/hmr work_dirs/hmr/latest.pth 8
```


## Training

### Training with a single / multiple GPUs

```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --no-validate
```
Example: using 1 GPU to train HMR.
```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --gpus 1 --no-validate
```

### Training with Slurm

If you can run MMHuman3D on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`.

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --no-validate
```

Common optional arguments include:
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--no-validate`: Whether not to evaluate the checkpoint during training.

Example: using 8 GPUs to train HMR on a slurm cluster.
```shell
./tools/slurm_train.sh my_partition my_job configs/hmr/resnet50_hmr_pw3d.py work_dirs/hmr 8 --no-validate
```

You can check [slurm_train.sh](https://github.com/open-mmlab/mmhuman3d/tree/main/tools/slurm_train.sh) for full arguments and environment variables.


## More Tutorials

- [Camera conventions](./cameras.md)
- [Keypoint conventions](./keypoints_convention.md)
- [Custom keypoint conventions](./customize_keypoints_convention.md)
- [HumanData](./human_data.md)
- [Keypoint visualization](./visualize_keypoints.md)
- [Mesh visualization](./visualize_smpl.md)
