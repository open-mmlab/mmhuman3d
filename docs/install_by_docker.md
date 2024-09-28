# Installation by Docker

After pull the image from [here](https://hub.docker.com/repository/docker/jacobbbbbb/mmhuman3d/general), simply run the following command:
```
docker run --gpus all --shm-size=64g -it -v {DATA_DIR}:/mmhuman3d/data mmhuman3d
```

We test this command on RTX 3090. Alternatively, you can build docker image based on your own GPU type by the following command with the provided Dockerfile:
```
docker image build -t mmhuman3d
```

Then run the command above. Note that inside the container, we donot provide data including body_models etc, please refer to [data_preparation](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/preprocess_dataset.md) for preparation details.

After data prepared, this [demo command](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/getting_started.md) can be a quick check:
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
