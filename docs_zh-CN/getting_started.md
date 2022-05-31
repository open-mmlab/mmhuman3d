# 开始使用

- [开始](#getting-started)
  - [安装](#installation)
  - [数据预处理](#data-preparation)
  - [人体模型](#body-model-preparation)
  - [推理 / 演示](#inference--demo)
    - [单人](#single-person)
    - [多人](#multi-person)
  - [测试](#evaluation)
    - [使用单（多）个GPU进行测试](#evaluate-with-a-single-gpu--multiple-gpus)
    - [使用slurm进行测试](#evaluate-with-slurm)
  - [训练](#training)
    - [使用单（多）个GPU进行训练](#training-with-a-single--multiple-gpus)
    - [通过slurm进行训练](#training-with-slurm)
  - [更多教程](#more-tutorials)

## 安装

安装请参考 [install.md](../docs_zh-CN/install.md).

## 数据预处理

数据预处理请参考 [data_preparation.md](../docs_zh-CN/preprocess_dataset.md).

## 人体模型

- 主要使用[SMPL](https://smpl.is.tue.mpg.de/) v1.0.
  - 可以在[SMPLify](https://smplify.is.tue.mpg.de/)下载neutral model.
  - 所有的人体模型需要重命名为 `SMPL_{GENDER}.pkl` 格式。 <br/>
    例如, `SMPL_NEUTRAL.pkl` `SMPL_MALE.pkl` `SMPL_FEMALE.pkl`
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)

下载上述所有文件，并按如下结构安排文件夹:

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

## 推理 / 演示
我们提供了用于从图像或视频中估计单人或多人`SMPL`模型参数的演示脚本，运行该脚本可能会用到`MMDetection`和`MMTracking`。通过该脚本进行演示，您只需要从模型库中选择用到的预训练模型(我们目前只支持[HMR](https://github.com/open-mmlab/mmhuman3d/tree/main/configs/hmr/)、 [SPIN](https://github.com/open-mmlab/mmhuman3d/tree/main/configs/spin/)和 [VIBE](https://github.com/open-mmlab/mmhuman3d/tree/main/configs/vibe/), 未来会加入更多的SOTA方法)，并指定少量的参数。

以下是一些参数的释义:

- 如果指定`--output` 和 `--show_path`, 演示脚本会将结果储存在`human_data`中，并且渲染得到的人体mesh。
- 如果指定`--smooth_type`, 演示脚本会使用特定的方法进行平滑。目前可以选择的平滑方法包括`guas1d`、`oneeuro`和 `savgol`。
- 如果指定`--speed_up_type`, 演示脚本会使用特定的方法进行加速处理。目前支持基于学习的方法`deciwatch`, 更多的信息请参考[DeciWatch](../configs/_base_/post_processing/README.md)。

### 单人

```shell
python demo/estimate_smpl.py \
    ${MMHUMAN3D_CONFIG_FILE} \
    ${MMHUMAN3D_CHECKPOINT_FILE} \
    --single_person_demo \
    --det_config ${MMDET_CONFIG_FILE} \
    --det_checkpoint ${MMDET_CHECKPOINT_FILE} \
    --input_path ${VIDEO_PATH_OR_IMG_PATH} \
    [--show_path ${VIS_OUT_PATH}] \
    [--output ${RESULT_OUT_PATH}] \
    [--smooth_type ${SMOOTH_TYPE}] \
    [--speed_up_type ${SPEED_UP_TYPE}] \
    [--draw_bbox] \
```

例如:
```shell
python demo/estimate_smpl.py \
    configs/hmr/resnet50_hmr_pw3d.py \
    data/checkpoints/resnet50_hmr_pw3d.pth \
    --single_person_demo \
    --det_config demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --input_path  demo/resources/single_person_demo.mp4 \
    --show_path vis_results/single_person_demo.mp4 \
    --output demo_result \
    --smooth_type savgol \
    --speed_up_type deciwatch \
    --draw_bbox
```

### 多人


```shell
python demo/estimate_smpl.py \
    ${MMHUMAN3D_CONFIG_FILE} \
    ${MMHUMAN3D_CHECKPOINT_FILE} \
    --multi_person_demo \
    --tracking_config ${MMTRACKING_CONFIG_FILE} \
    --input_path ${VIDEO_PATH_OR_IMG_PATH} \
    [--show_path ${VIS_OUT_PATH}] \
    [--output ${RESULT_OUT_PATH}] \
    [--smooth_type ${SMOOTH_TYPE}] \
    [--speed_up_type ${SPEED_UP_TYPE}] \
    [--draw_bbox]
```

例如:
```shell
python demo/estimate_smpl_image.py \
    configs/hmr/resnet50_hmr_pw3d.py \
    data/checkpoints/resnet50_hmr_pw3d.pth \
    --multi_person_demo \
    --tracking_config demo/mmtracking_cfg/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py \
    --input_path  demo/resources/multi_person_demo.mp4 \
    --show_path vis_results/multi_person_demo.mp4 \
    --smooth_type savgol \
    --speed_up_type deciwatch \
    [--draw_bbox]

```
MMHuman3D的checkpoints文件可以从[model zoo](../docs_zh-CN/model_zoo.md)下载。
这里我们使用HMR (resnet50_hmr_pw3d.pth)作为示例。
## 测试

我们提供的预训练模型可以参见 [config](https://github.com/open-mmlab/mmhuman3d/tree/main/configs).

### 使用单（多）个GPU进行测试

```shell
python tools/test.py ${CONFIG} --work-dir=${WORK_DIR} ${CHECKPOINT} --metrics=${METRICS}
```
例如:
```shell
python tools/test.py configs/hmr/resnet50_hmr_pw3d.py --work-dir=work_dirs/hmr work_dirs/hmr/latest.pth --metrics pa-mpjpe mpjpe
```

### 使用slurm进行测试

如果通过[slurm](https://slurm.schedmd.com/)在集群上使用MMHuman3D, 您可以使用脚本 `slurm_test.sh`进行测试。

```shell
./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${WORK_DIR} ${CHECKPOINT} --metrics ${METRICS}
```
例如:
```shell
./tools/slurm_test.sh my_partition test_hmr configs/hmr/resnet50_hmr_pw3d.py work_dirs/hmr work_dirs/hmr/latest.pth 8 --metrics pa-mpjpe mpjpe
```


## 训练

### 使用单（多）个GPU进行训练

```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --no-validate
```
例如: 使用单个GPU训练HMR.
```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --gpus 1 --no-validate
```

### 使用slurm进行训练

如果通过[slurm](https://slurm.schedmd.com/)在集群上使用MMHuman3D, 您可以使用脚本 `slurm_train.sh`进行训练。

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --no-validate
```

可选参数包括:
- `--resume-from ${CHECKPOINT_FILE}`: 继承之前的权重继续训练。
- `--no-validate`: 在训练时是否进行测试。

例如: 在集群上使用8卡训练HMR。
```shell
./tools/slurm_train.sh my_partition my_job configs/hmr/resnet50_hmr_pw3d.py work_dirs/hmr 8 --no-validate
```

详情请见[slurm_train.sh](https://github.com/open-mmlab/mmhuman3d/tree/main/tools/slurm_train.sh)。

## 更多教程

- [Camera conventions](../docs_zh-CN/cameras.md)
- [Keypoint conventions](../docs_zh-CN/keypoints_convention.md)
- [Custom keypoint conventions](../docs_zh-CN/customize_keypoints_convention.md)
- [HumanData](../docs_zh-CN/human_data.md)
- [Keypoint visualization](../docs_zh-CN/visualize_keypoints.md)
- [Mesh visualization](../docs_zh-CN/visualize_smpl.md)
