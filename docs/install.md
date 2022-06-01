# Installation

<!-- TOC -->

- [Requirements](#requirements)
- [Prepare environment](#prepare-environment)
- [Install MMHuman3D](#install-mmhuman3d)
- [A from-scratch setup script](#a-from-scratch-setup-script)

<!-- TOC -->

## Requirements

- Linux
- ffmpeg
- Python 3.7+
- PyTorch 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0 or 1.9.1.
- CUDA 9.2+
- GCC 5+
- PyTorch3D 0.4+
- [MMCV](https://github.com/open-mmlab/mmcv) (Please install mmcv-full>=1.3.17,<1.6.0 for GPU)

Optional:
- [MMPOSE](https://github.com/open-mmlab/mmpose) (Only for demo.)
- [MMDETECTION](https://github.com/open-mmlab/mmdetection) (Only for demo.)
- [MMTRACKING](https://github.com/open-mmlab/mmtracking) (Only for multi-person demo. If you use mmtrack, please install mmcls<1.18.0, mmcv-full>=1.3.16,<1.6.0 for GPU

## Prepare environment

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
```

b. Install ffmpeg

Install ffmpeg with conda directly and the libx264 will be built automatically.

```shell
conda install ffmpeg
```

c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).
```shell
conda install pytorch={torch_version} torchvision cudatoolkit={cu_version} -c pytorch
```

E.g., install PyTorch 1.8.0 & CUDA 10.2.
```shell
conda install pytorch=1.8.0 torchvision cudatoolkit=10.2 -c pytorch
```

**Important:** Make sure that your compilation CUDA version and runtime CUDA version match.

d. Install PyTorch3D and dependency libs.

```shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y

conda install pytorch3d -c pytorch3d
```
Please refer to [PyTorch3D-install](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for details.

Your installation is successful if you can do these in command line.

```shell
echo "import pytorch3d;print(pytorch3d.__version__); \
    from pytorch3d.renderer import MeshRenderer;print(MeshRenderer);\
    from pytorch3d.structures import Meshes;print(Meshes);\
    from pytorch3d.renderer import cameras;print(cameras);\
    from pytorch3d.transforms import Transform3d;print(Transform3d);"|python

echo "import torch;device=torch.device('cuda');\
    from pytorch3d.utils import torus;\
    Torus = torus(r=10, R=20, sides=100, rings=100, device=device);\
    print(Torus.verts_padded());"|python
```

## Install MMHuman3D

a. Build mmcv-full & mmpose & mmdet & mmtrack

- mmcv-full

We recommend you to install the pre-build package as below.

For CPU:
```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/{torch_version}/index.html
```
Please replace `{torch_version}` in the url to your desired one.

For GPU:
 ```shell
 pip install "mmcv-full>=1.3.17,<1.6.0" -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
 ```
Please replace `{cu_version}` and `{torch_version}` in the url to your desired one.

For example, to install mmcv-full with CUDA 10.2 and PyTorch 1.8.0, use the following command:
```shell
pip install "mmcv-full>=1.3.17,<1.6.0" -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
```

See [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for different versions of MMCV compatible to different PyTorch and CUDA versions.
For more version download link, refer to [openmmlab-download](https://download.openmmlab.com/mmcv/dist/index.html).

Optionally you can choose to compile mmcv from source by the following command

```shell
git clone https://github.com/open-mmlab/mmcv.git -b v1.3.17
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
# OR pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
cd ..
```

Important: You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

- mmdetection (optional)

```shell
pip install mmdet
```

Optionally, you can also build MMDetection from source in case you want to modify the code:
```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

- mmpose (optional)
```shell
pip install mmpose
```

Optionally, you can also build MMPose from source in case you want to modify the code:

```shell
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```

- mmtracking (optional)

```shell
pip install "mmcls<0.18.0" "mmtrack<0.9.0,>=0.8.0"
```

Optionally, you can also build MMTracking from source in case you want to modify the code:

```shell
git clone git@github.com:open-mmlab/mmtracking.git -b v0.8.0
cd mmtracking
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```
b. Clone the mmhuman3d repository.

```shell
git clone https://github.com/open-mmlab/mmhuman3d.git
cd mmhuman3d
```


c. Install build requirements and then install mmhuman3d.

```shell
pip install -v -e .  # or "python setup.py develop"
```

## A from-scratch setup script

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab

conda install pytorch==1.8.0 torchvision cudatoolkit=10.2 -c pytorch -y

# install PyTorch3D

conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y

conda install pytorch3d -c pytorch3d

# install mmcv-full

pip install "mmcv-full>=1.3.17,<1.6.0" -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html


# Optional
# install mmdetection & mmpose & mmtracking

pip install mmdet

pip install mmpose

pip install "mmcls<0.18.0" "mmtrack<0.9.0,>=0.8.0"

# install mmhuman3d

git clone https://github.com/open-mmlab/mmhuman3d.git
cd mmhuman3d
pip install -v -e .
```
