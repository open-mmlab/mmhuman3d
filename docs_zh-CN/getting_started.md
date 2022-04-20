# 开始使用

* [安装](#installation)
* [数据预处理](#data-preparation)
* [人体模型](#body-model-preparation)
* [推理 / Demo](#inference--demo)
  + [单人](#single-person)
  + [多人](#multi-person)
* [验证](#evaluation)
  + [使用单（多）个GPU进行验证](#evaluate-with-a-single-gpu--multiple-gpus)
  + [使用slurm进行验证](#evaluate-with-slurm)
* [训练](#training)
  + [使用单（多）个GPU进行训练](#training-with-a-single--multiple-gpus)
  + [通过slurm进行训练](#training-with-slurm)
* [更多教程](#more-tutorials)

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

## 推理 / Demo

```shell
python demo/estimate_smpl_image.py ${CONFIG_FILE} ${CHECKPOINT} [optional]
```

### 单人
可选参数`[optional]`包括:
- `--single_person_demo`: 单人推理
- `--det_config`: MMDetection的config文件
- `--det_checkpoint`: MMDetection的checkpoint文件
- `--input_path`: 输入的路径
- `--show_path`: 渲染后的图片或视频的保存路径
- `--smooth_type`: 平滑模式

例如:
```shell
python demo/estimate_smpl_image.py \
    configs/hmr/resnet50_hmr_pw3d.py \
    data/checkpoints/resnet50_hmr_pw3d.pth \
    --single_person_demo \
    --det_config demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --input_path  demo/resources/single_person_demo.mp4 \
    --show_path vis_results/single_person_demo.mp4 \
    --smooth_type savgol
```

注意MMHuman3D的checkpoints文件可以从[model zoo](model_zoo.md)下载。
这里我们使用HMR (resnet50_hmr_pw3d.pth)作为样例。

### 多人
可选参数`[optional]`包括:
- `--multi_person_demo`: 多人推理
- `--mmtracking_config`: MMTracking的config文件
- `--input_path`: 输入的路径
- `--show_path`: 渲染后的图片或视频的保存路径
- `--smooth_type`: 平滑模式

例如:
```shell
python demo/estimate_smpl_image.py \
    configs/hmr/resnet50_hmr_pw3d.py \
    data/checkpoints/resnet50_hmr_pw3d.pth \
    --multi_person_demo \
    --tracking_config demo/mmtracking_cfg/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py \
    --input_path  demo/resources/multi_person_demo.mp4 \
    --show_path vis_results/multi_person_demo.mp4 \
    --smooth_type savgol
```

## 验证

我们提供的预训练模型可以参见 [config](https://github.com/open-mmlab/mmhuman3d/tree/main/configs).

### 使用单（多）个GPU进行验证

```shell
python tools/test.py ${CONFIG} --work-dir=${WORK_DIR} ${CHECKPOINT} --metrics=${METRICS}
```
例如:
```shell
python tools/test.py configs/hmr/resnet50_hmr_pw3d.py --work-dir=work_dirs/hmr work_dirs/hmr/latest.pth --metrics pa-mpjpe mpjpe
```

### 使用slurm进行验证

如果你通过[slurm](https://slurm.schedmd.com/)在集群上使用MMHuman3D, 你可以使用脚本 `slurm_test.sh`进行验证。

```shell
./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${WORK_DIR} ${CHECKPOINT} --metrics ${METRICS}
```
例如:
```shell
./tools/slurm_test.sh my_partition test_hmr configs/hmr/resnet50_hmr_pw3d.py work_dirs/hmr work_dirs/hmr/latest.pth 8 --metrics pa-mpjpe mpjpe
```


## Training

### 使用单（多）个GPU进行训练

```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --no-validate
```
例如: 使用单个GPU训练HMR.
```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --gpus 1 --no-validate
```

### 使用slurm进行训练

如果你通过[slurm](https://slurm.schedmd.com/)在集群上使用MMHuman3D, 你可以使用脚本 `slurm_train.sh`进行训练。

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --no-validate
```

可选参数包括:
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--no-validate`: 在训练时是否进行验证.

例如: 在集群上使用8卡训练HMR。
```shell
./tools/slurm_train.sh my_partition my_job configs/hmr/resnet50_hmr_pw3d.py work_dirs/hmr 8 --no-validate
```

详情请见[slurm_train.sh](https://github.com/open-mmlab/mmhuman3d/tree/main/tools/slurm_train.sh)。

## 更多教程

- [Camera conventions](./cameras.md)
- [Keypoint conventions](./keypoints_convention.md)
- [Custom keypoint conventions](./customize_keypoints_convention.md)
- [HumanData](./human_data.md)
- [Keypoint visualization](./visualize_keypoints.md)
- [Mesh visualization](./visualize_smpl.md)