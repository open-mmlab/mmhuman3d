<div align="center">
    <img src="resources/mmhuman3d-logo.png" width="600"/>
</div>

<!-- [![Documentation](https://readthedocs.org/projects/mmhuman3d/badge/?version=latest)](https://mmhuman3d.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmhuman3d/workflows/build/badge.svg)](https://github.com/open-mmlab/mmhuman3d/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmhuman3d/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmhuman3d)
[![PyPI](https://img.shields.io/pypi/v/mmhuman3d)](https://pypi.org/project/mmhuman3d/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmhuman3d.svg)](https://github.com/open-mmlab/mmhuman3d/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmhuman3d.svg)](https://github.com/open-mmlab/mmhuman3d/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmhuman3d.svg)](https://github.com/open-mmlab/mmhuman3d/issues) -->

[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmhuman3d.svg)](https://github.com/open-mmlab/mmhuman3d/blob/master/LICENSE)

## 简介

[English](README.md) | 简体中文

MMHuman3D 是一款基于 PyTorch 的人体参数化模型的开源工具箱，是 [OpenMMLab](https://openmmlab.com/) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.7** 以上的版本。

https://user-images.githubusercontent.com/62529255/144362861-e794b404-c48f-4ebe-b4de-b91c3fbbaa3b.mp4

### 主要特性

- **使用模块化的框架复现流行的算法**

  MMHuman3D 重新实现了流行的算法，帮助用户只需一行代码即可完成复现。模块化的框架适合快速验证：用户可以在不修改代码的情况下调试不同的超参数甚至神经网络结构。

- **通过一个统一的数据规范 HumanData 支持多种数据集**

  通过一个规范工具箱的帮助，我们将所有的支持的数据集都对齐到一个统一的数据格式 *HumanData* . 我们同时也提供预处理完成的数据文件。

- **多功能可视化工具箱**

  一整套可微的可视化工具支持人体参数化模型的渲染（包括部分分割，深度图以及点云）和传统 2D/3D 关键点的可视化。

## 基准与模型库

更多详情可见 [模型库](docs/model_zoo.md)。

已支持的算法：

<details open>
<summary>(click to collapse)</summary>

- [x] SMPLify (ECCV'2016)
- [x] SMPLify-X (CVPR'2019)
- [x] HMR (CVPR'2018)
- [x] SPIN (ICCV'2019)
- [x] VIBE (CVPR'2020)
- [x] HybrIK (CVPR'2021)

</details>

已支持的数据集：

<details open>
<summary>(click to collapse)</summary>

- [x] 3DPW (ECCV'2018)
- [x] AGORA (CVPR'2021)
- [x] AMASS (ICCV'2019)
- [x] COCO (ECCV'2014)
- [x] COCO-WholeBody (ECCV'2020)
- [x] CrowdPose (CVPR'2019)
- [x] EFT (3DV'2021)
- [x] Human3.6M (TPAMI'2014)
- [x] InstaVariety (CVPR'2019)
- [x] LSP (BMVC'2010)
- [x] LSP-Extended (CVPR'2011)
- [x] MPI-INF-3DHP (3DC'2017)
- [x] MPII (CVPR'2014)
- [x] Penn Action (ICCV'2012)
- [x] PoseTrack18 (CVPR'2018)
- [x] SURREAL (CVPR'2017)
- [x] UP3D (CVPR'2017)

</details>

我们将跟进学界的最新进展，并支持更多算法和框架。

如果您对MMHuman3D有任何功能需求，请随时在[愿望清单](https://github.com/open-mmlab/mmhuman3d/issues/3)中留言。


## 快速入门

请参考[快速入门](docs/getting_started.md)文档学习 MMDetection 的基本使用。

## 许可

该项目采用 [Apache 2.0 license](LICENSE) 开源协议。部分支持的算法可能采用了[额外协议](docs/additional_licenses.md)。

## 引用

如果您觉得 MMHuman3D 对您的研究有所帮助，请考虑引用它：

```bibtex
@misc{mmhuman3d,
    title={OpenMMLab Human Pose and Shape Estimation Toolbox and Benchmark},
    author={MMHuman3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmhuman3d}},
    year={2021}
}
```

## 参与贡献

我们非常欢迎用户对于 MMHuman3D 做出的任何贡献，可以参考 [CONTRIBUTION.md](.github/CONTRIBUTING.md) 文件了解更多细节。

## 致谢

MMHuman3D是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## OpenMMLab的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱与测试基准
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用3D目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 新一代生成模型工具箱
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d)：OpenMMLab 人体参数化模型工具箱与测试基准

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=GJP18SjI)

<div align="center">
<img src="docs/imgs/zhihu_qrcode.jpg" height="400" />  <img src="docs/imgs/qq_group2_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
