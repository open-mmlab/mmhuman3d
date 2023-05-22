ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda install ffmpeg -y
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
RUN conda install -c bottler nvidiacub -y
RUN conda install pytorch3d -c pytorch3d -y
RUN pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
RUN pip install "mmdet<=2.25.1"
WORKDIR /
RUN git clone https://github.com/open-mmlab/mmhuman3d.git
WORKDIR /mmhuman3d
RUN pip install -v -e .

RUN pip install numpy==1.23
RUN pip install pillow==9.5
RUN pip install scipy==1.9.1
