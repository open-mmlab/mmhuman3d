# Learning-Based Post-Processing methods

We provide two learning-based post-processing methods ```deciwatch``` and ```smoothnet``` for speed-up and smoothing respectively.

## DeciWatch

We provide the config files for DeciWatch: [DeciWatch: A Simple Baseline for 10x Efficient 2D and 3D Pose Estimation](https://arxiv.org/pdf/2203.08713.pdf).

```BibTeX
@article{zeng2022deciwatch,
  title={DeciWatch: A Simple Baseline for 10x Efficient 2D and 3D Pose Estimation},
  author={Zeng, Ailing and Ju, Xuan and Yang, Lei and Gao, Ruiyuan and Zhu, Xizhou and Dai, Bo and Xu, Qiang},
  journal={arXiv preprint arXiv:2203.08713},
  year={2022}
}
```

### Notes

We use checkpoints trained on SPIN-3DPW for demo speed up. Checkpoints with different intervals and q values are provided. If you need more checkpoints trained on various datasets and backbones, please refer to the [official implementation of DeciWatch](https://github.com/cure-lab/DeciWatch).


| Interval |  Window Q  |Config  | Download | Speed Up | Precision Improvement (MPJPE In/Out) |
|:------:|:-------:|:------:|:------:|:------:|:------:|
| 10 | 1 |[deciwatch_interval10_q1](deciwatch_interval10_q1.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/deciwatch/deciwatch_interval10_q1.pth.tar?versionId=CAEQOhiBgMChhsS9gxgiIDM5OGUwZGY0MTc4NTQ2M2NhZDEwMzU5MWUzMWNmZjY1)| 10X |99.35 / 95.85|
| 10 | 2 |[deciwatch_interval10_q2](deciwatch_interval10_q2.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/deciwatch/deciwatch_interval10_q2.pth.tar?versionId=CAEQOhiBgICau8O9gxgiIDk1Y2Y0MzUxMmY0MDQzZThiYzhkMGJlMjc3ZDQ2NTQ2)| 10X | 99.45 / 96.37|
| 10 | 3 |[deciwatch_interval10_q3](deciwatch_interval10_q3.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/deciwatch/deciwatch_interval10_q3.pth.tar?versionId=CAEQOhiBgICIq8O9gxgiIDZiMjEzMjY3ODA4MTQwNGY5NTU3OWNkZjRjZjI2ZDFi)| 10X | 99.60 / 96.98 |
| 10 | 4 |[deciwatch_interval10_q4](deciwatch_interval10_q4.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/deciwatch/deciwatch_interval10_q4.pth.tar?versionId=CAEQOhiBgICUq8O9gxgiIDJkZjUwYWJmNTRkNjQxMDE4YmUyNWMwNTcwNGQ4M2Ix)| 10X | 99.58 / 96.87 |
| 10 | 5 |[deciwatch_interval10_q5](deciwatch_interval10_q5.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/deciwatch/deciwatch_interval10_q5.pth.tar?versionId=CAEQOhiBgMCN7MS9gxgiIDUwNGFhM2Y0MGI3MjRiYWQ5NzZjODMwMDk3ZjU1OTk3)| 10X | 99.78 / 97.39 |
| 5 | 1 |[deciwatch_interval5_q1](deciwatch_interval5_q1.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/deciwatch/deciwatch_interval5_q1.pth.tar?versionId=CAEQOhiBgIDfocS9gxgiIDkxN2Y3OWQzZmJiMTQyMTM5NWZhZTYxYmI0MDlmMDBh) | 5X |99.31 / 95.05 |
| 5 | 2 |[deciwatch_interval5_q2](deciwatch_interval5_q2.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/deciwatch/deciwatch_interval5_q2.pth.tar?versionId=CAEQOhiBgIDgu8O9gxgiIDNjMDEyOWQ3NjRkODQ2YTI5MjUxYWU4NzhjOTc1YTRj) | 5X | 99.35 / 95.05 |
| 5 | 3 |[deciwatch_interval5_q3](deciwatch_interval5_q3.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/deciwatch/deciwatch_interval5_q3.pth.tar?versionId=CAEQOhiBgIDJs8O9gxgiIDk1MDExMjI5Y2U1MDRmZjViMDBjOGU5YzY3OTRlNmE5) | 5X | 99.45 / 94.84 |
| 5 | 4 |[deciwatch_interval5_q4](deciwatch_interval5_q4.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/deciwatch/deciwatch_interval5_q4.pth.tar?versionId=CAEQOhiBgMC.t8O9gxgiIGZjZWY3OTdhNGRjZjQyNjY5MGU5YzkxZTZjMWU1MTA2) |5X | 99.45 / 94.94 |
| 5 | 5 |[deciwatch_interval5_q5](deciwatch_interval5_q5.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/deciwatch/deciwatch_interval5_q5.pth.tar?versionId=CAEQOhiBgMCyq8O9gxgiIDRjMzViMjllNWRiNjRlMzA5ZjczYWIxOGU2OGFkYjdl) |5X | 99.55 / 94.48 |

To use different settings of DeciWatch in demo, specify `--speed_up_type` with the checkpoint name. For example, you may use `--speed_up_type deciwatch_interval10_q3` for 10X speed up with a window size of 31. Simply set `--speed_up_type deciwatch` to use default setting `deciwatch_interval5_q3`. The meaning of interval and q can be found in the original paper.

## SmoothNet

We provide the config files for [SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos](https://arxiv.org/abs/2112.13715).


```BibTeX
@article{zeng2021smoothnet,
  title={SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos},
  author={Zeng, Ailing and Yang, Lei and Ju, Xuan and Li, Jiefeng and Wang, Jianyi and Xu, Qiang},
  journal={arXiv preprint arXiv:2112.13715},
  year={2021}
}
```

### Notes

We use checkpoints trained on SPIN-3DPW for demo pose smoothing. Checkpoints with different window size are provided.


| Window Size  |Config  | Download | Precision Improvement (MPJPE In/Out) | Smoothness Improvement (Accel In/Out) |
|:------:|:-------:|:------:|:------:|:------:|
| 8 |[smoothnet_windowsize8](smoothnet_windowsize8.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smoothnet/smoothnet_windowsize8.pth.tar?versionId=CAEQPhiBgMDo0s7shhgiIDgzNTRmNWM2ZWEzYTQyYzRhNzUwYTkzZWZkMmU5MWEw)| 96.85 / 95.84 | 34.62 / 7.13 |
| 16 |[smoothnet_windowsize16](smoothnet_windowsize16.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smoothnet/smoothnet_windowsize16.pth.tar?versionId=CAEQPhiBgMC.s87shhgiIGM3ZTI1ZGY1Y2NhNDQ2YzRiNmEyOGZhY2VjYWFiN2Zi)| 96.85 / 95.61| 34.62 / 6.35 |
| 32 |[smoothnet_windowsize32](smoothnet_windowsize32.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smoothnet/smoothnet_windowsize32.pth.tar?versionId=CAEQPhiBgIDf0s7shhgiIDhmYmM3YWQ0ZGI3NjRmZTc4NTk2NDE1MTA2MTUyMGRm)| 96.85 / 95.03 |34.62 / 6.11 |
| 64 |[smoothnet_windowsize64](smoothnet_windowsize64.py)| [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smoothnet/smoothnet_windowsize64.pth.tar?versionId=CAEQPhiBgMCyw87shhgiIGEwODI4ZjdiYmFkYTQ0NzZiNDVkODk3MDBlYzE1Y2Rh)| 96.85 / 95.26 | 34.62 / 6.02 |

To use different settings of SmoothNet in demo, specify `--smooth_type` with the checkpoint name. For example, you may use `--smooth_type smoothnet_windowsize8` for pose smoothing with a window size of 8. Simply set `--mooth_type smoothnet` to use default setting `smoothnet_windowsize8`. The meaning of windowsize can be found in the original paper.
