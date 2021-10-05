conda create -n mmhuman3d python=3.7
source activate mmhuman3d

# boost
export CPLUS_INCLUDE_PATH=/mnt/lustre/share/boost/include/boost:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/mnt/lustre/share/boost/include/boost:$C_INCLUDE_PATH
export BOOST_INCLUDE_DIRS=/mnt/lustre/share/boost/include/boost

# gcc-5.4
export CXX=/mnt/lustre/share/caizhongang/gcc-5.4.0/install/gcc-5.4/bin/g++
export CC=/mnt/lustre/share/caizhongang/gcc-5.4.0/install/gcc-5.4/bin/gcc
export GCC=/mnt/lustre/share/caizhongang/gcc-5.4.0/install/gcc-5.4/bin/gcc
export PATH=/mnt/lustre/share/caizhongang/gcc-5.4.0/install/gcc-5.4/bin:/mnt/lustre/share/caizhongang/gcc-5.4.0/install/gcc-5.4/lib64:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/caizhongang/gcc-5.4.0/install/gcc-5.4/lib:$LD_LIBRARY_PATH

# cmake >= 3.13.2 is required
export PATH=/mnt/lustre/share/caizhongang/cmake-3.17.0-Linux-x86_64/bin:$PATH

# cuda paths
export PATH=/mnt/lustre/share/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-10.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/mnt/lustre/share/cuda-10.1

conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c bottler nvidiacub
conda install -y -c pytorch3d pytorch3d

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/101/torch1.7.1/index.html
pip install chumpy
pip install smplx
pip install scikit-image
