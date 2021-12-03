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

## Introduction

MMHuman3D is an open source human pose and shape estimation toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The main branch works with **PyTorch 1.7+**.

https://user-images.githubusercontent.com/62529255/144362861-e794b404-c48f-4ebe-b4de-b91c3fbbaa3b.mp4

### Major Features

- **Reproducing popular methods with a modular framework**

  MMHuman3D reimplements popular methods, allowing users to reproduce SOTAs with one line of code. The modular framework is convenient for rapid prototyping: the users may attempt various hyperparameter settings and even network architectures, without actually modifying the code.

- **Supporting various datasets with a unified data convention**

  With the help of a convention toolbox, a unified data format *HumanData* is used to align all supported datasets. Preprocessed data files are also available.

- **Versatile visualization toolbox**

  A suite of differentiale visualization tools for human parametric model rendering (including part segmentation, depth map and point clouds) and conventional 2D/3D keypoints are available.

## Benchmark and Model Zoo

More details can be found in [model_zoo.md](docs/model_zoo.md).

Supported methods:

<details open>
<summary>(click to collapse)</summary>

- [x] SMPLify (ECCV'2016)
- [x] SMPLify-X (CVPR'2019)
- [x] HMR (CVPR'2018)
- [x] SPIN (ICCV'2019)
- [x] VIBE (CVPR'2020)
- [x] HybrIK (CVPR'2021)

</details>

Supported datasets:

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

We will keep up with the latest progress of the community, and support more popular methods and frameworks.

If you have any feature requests, please feel free to leave an issue.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMHuman3D.

## License

This project is released under the [Apache 2.0 license](LICENSE). Some supported methods may carry [additional licenses](docs/additional_licenses.md).

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmhuman3d,
    title={OpenMMLab Human Pose and Shape Estimation Toolbox and Benchmark},
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
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab Human Pose and Shape Estimation Toolbox and Benchmark.
