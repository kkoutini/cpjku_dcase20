#!/bin/bash

ENVNAME=${1:-cpjku_dcase20}


echo "Please be patient, $ENVNAME conda environment will be created and dependencies will be installed"

conda config --add channels conda-forge


conda create -y -n $ENVNAME python=3.7 scipy pandas h5py cython numpy pyyaml mkl setuptools cmake cffi  tqdm librosa ffmpeg tensorflow mkl-include typing opencv

source activate $ENVNAME




#conda install -y cuda91 pytorch torchvision  -c pytorch
#conda install -y tensorflow-gpu-base
#conda install -y -c anaconda tensorflow-gpu
#conda install -y -c anaconda
pip install sed_eval pynvrtc dcase_util attrdict GitPython pymongo torch==1.5 IPython


# tensorflow gpu, go to: https://www.tensorflow.org/install


pip install git+https://github.com/lanpa/tensorboard-pytorch


##############
# install building pytorch dependencies
# Install basic dependencies
#conda install -y
#conda install -y -c mingfeima mkldnn

# Add LAPACK support for the GPU
#conda install -c pytorch magma-cuda91

###############

echo Clean/remove environment using :
echo $ source deactivate
echo $ conda env remove -n $ENVNAME
echo remember always \"source activate $ENVNAME\"


