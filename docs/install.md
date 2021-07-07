## Installation

<!-- TOC -->

- [Requirements](#requirements)

<!-- TOC -->

### Requirements

- Linux
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+
- GCC 5+
- [MMCV](https://github.com/open-mmlab/mmcv)

### Install MMHuman3D

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

c. Clone the mmhuman3d repository.

```shell
git clone git@gitlab.bj.sensetime.com:zeotrope/zoehuman/mmhuman3d.git
cd zeomotion
```

d. Install build requirements and then install mmhuman3d.

```shell
pip install -e .  # or "python setup.py develop"
```
