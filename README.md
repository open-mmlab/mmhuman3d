<br/>

<div align="center">
    <img src="resources/mmhuman3d-logo.png" width="600"/>
</div>

<br/>

<div align="center">

[![Documentation](https://readthedocs.org/projects/mmhuman3d/badge/?version=latest)](https://mmhuman3d.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmhuman3d/workflows/build/badge.svg)](https://github.com/open-mmlab/mmhuman3d/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmhuman3d/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmhuman3d)
[![PyPI](https://img.shields.io/pypi/v/mmhuman3d)](https://pypi.org/project/mmhuman3d/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmhuman3d.svg)](https://github.com/open-mmlab/mmhuman3d/blob/main/LICENSE)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmhuman3d.svg)](https://github.com/open-mmlab/mmhuman3d/issues)

</div>

## Introduction

English | [简体中文](README_CN.md)

MMHuman3D is an open source PyTorch-based codebase for the use of 3D human parametric models in computer vision and computer graphics. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The main branch works with **PyTorch 1.7+**.

https://user-images.githubusercontent.com/62529255/144362861-e794b404-c48f-4ebe-b4de-b91c3fbbaa3b.mp4

### Major Features

- **Reproducing popular methods with a modular framework**

  MMHuman3D reimplements popular methods, allowing users to reproduce SOTAs with one line of code. The modular framework is convenient for rapid prototyping: the users may attempt various hyperparameter settings and even network architectures, without actually modifying the code.

- **Supporting various datasets with a unified data convention**

  With the help of a convention toolbox, a unified data format *HumanData* is used to align all supported datasets. Preprocessed data files are also available.

- **Versatile visualization toolbox**

  A suite of differentiale visualization tools for human parametric model rendering (including part segmentation, depth map and point clouds) and conventional 2D/3D keypoints are available.

## News
- 2022-07-08: MMHuman3D [v0.9.0](https://github.com/open-mmlab/mmhuman3d/releases/tag/v0.9.0) is released. Major updates include:
  - Support SMPL-X estimation with [ExPose](https://expose.is.tue.mpg.de/) for simultaneous recovery of face, hands and body
  - Support new body model [STAR](https://star.is.tue.mpg.de/)
  - Release of [GTA-Human](https://caizhongang.github.io/projects/GTA-Human/) dataset with SPIN-FT (51.98 mm) and PARE-FT (46.84 mm) baselines! (Official)
  - Refactor registration and improve performance of SPIN to 57.54 mm
- 2022-05-31: MMHuman3D [v0.8.0](https://github.com/open-mmlab/mmhuman3d/releases/tag/v0.8.0) is released. Major updates include:
  - Support SmoothNet (added by paper authors)
  - Fix circular import and up to 2.5x speed up in module initialization
  - Add documentations in Chinese
- 2022-04-30: MMHuman3D [v0.7.0](https://github.com/open-mmlab/mmhuman3d/releases/tag/v0.7.0) is released. Major updates include:
  - Support PARE (better than the official implementation)
  - Support DeciWatch (added by paper authors)
  - Add GTA-Human HMR baseline (official release)
  - Support saving inference results

## Benchmark and Model Zoo

More details can be found in [model_zoo.md](docs/model_zoo.md).

Supported body models:

<details open>
<summary>(click to collapse)</summary>

- [x] [SMPL](https://smpl.is.tue.mpg.de/) (SIGGRAPH Asia'2015)
- [x] [SMPL-X](https://smpl-x.is.tue.mpg.de/) (CVPR'2019)
- [x] [MANO](https://mano.is.tue.mpg.de/) (SIGGRAPH ASIA'2017)
- [x] [FLAME](https://flame.is.tue.mpg.de/) (SIGGRAPH ASIA'2017)
- [x] [STAR](https://star.is.tue.mpg.de/) (ECCV'2020)

</details>

Supported methods:

<details open>
<summary>(click to collapse)</summary>

- [x] [SMPLify](https://smplify.is.tue.mpg.de/) (ECCV'2016)
- [x] [SMPLify-X](https://smpl-x.is.tue.mpg.de/) (CVPR'2019)
- [x] [HMR](https://akanazawa.github.io/hmr/) (CVPR'2018)
- [x] [SPIN](https://www.seas.upenn.edu/~nkolot/projects/spin/) (ICCV'2019)
- [x] [VIBE](https://github.com/mkocabas/VIBE) (CVPR'2020)
- [x] [HybrIK](https://jeffli.site/HybrIK/) (CVPR'2021)
- [x] [PARE](https://pare.is.tue.mpg.de/) (ICCV'2021)
- [x] [DeciWatch](https://ailingzeng.site/deciwatch) (ECCV'2022)
- [x] [SmoothNet](https://ailingzeng.site/smoothnet) (ECCV'2022)
- [x] [ExPose](https://expose.is.tue.mpg.de) (ECCV'2020)
- [x] [BalancedMSE](https://sites.google.com/view/balanced-mse/home) (CVPR'2022)

</details>

Supported datasets:

<details open>
<summary>(click to collapse)</summary>

- [x] [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) (ECCV'2018)
- [x] [AGORA](https://agora.is.tue.mpg.de/) (CVPR'2021)
- [x] [AMASS](https://amass.is.tue.mpg.de/) (ICCV'2019)
- [x] [COCO](https://cocodataset.org/#home) (ECCV'2014)
- [x] [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody) (ECCV'2020)
- [x] [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) (CVPR'2019)
- [x] [EFT](https://github.com/facebookresearch/eft) (3DV'2021)
- [x] [GTA-Human](https://caizhongang.github.io/projects/GTA-Human/) (arXiv'2021)
- [x] [Human3.6M](http://vision.imar.ro/human3.6m/description.php) (TPAMI'2014)
- [x] [InstaVariety](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md) (CVPR'2019)
- [x] [LSP](https://sam.johnson.io/research/lsp.html) (BMVC'2010)
- [x] [LSP-Extended](https://sam.johnson.io/research/lspet.html) (CVPR'2011)
- [x] [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) (3DC'2017)
- [x] [MPII](http://human-pose.mpi-inf.mpg.de/) (CVPR'2014)
- [x] [Penn Action](http://dreamdragon.github.io/PennAction/) (ICCV'2012)
- [x] [PoseTrack18](https://posetrack.net/users/download.php) (CVPR'2018)
- [x] [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/) (CVPR'2017)
- [x] [UP3D](https://files.is.tuebingen.mpg.de/classner/up/) (CVPR'2017)
- [x] [FreiHand](https://lmb.informatik.uni-freiburg.de/projects/freihand/) (ICCV'2019)
- [x] [EHF](https://smpl-x.is.tue.mpg.de/) (CVPR'2019)
- [x] [Stirling/ESRC-Face3D](http://pics.psych.stir.ac.uk/ESRC/index.htm) (FG'2018)

</details>

We will keep up with the latest progress of the community, and support more popular methods and frameworks.

If you have any feature requests, please feel free to leave a comment in the [wishlist](https://github.com/open-mmlab/mmhuman3d/discussions/47).

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMHuman3D.

## License

This project is released under the [Apache 2.0 license](LICENSE). Some supported methods may carry [additional licenses](docs/additional_licenses.md).

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmhuman3d,
    title={OpenMMLab 3D Human Parametric Model Toolbox and Benchmark},
    author={MMHuman3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmhuman3d}},
    year={2021}
}
```

## Contributing

We appreciate all contributions to improve MMHuman3D. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMHuman3D is an open source project that is contributed by researchers and engineers from both the academia and the industry.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab next-generation toolbox for generative models.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab FewShot Learning Toolbox and Benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D Human Parametric Model Toolbox and Benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
