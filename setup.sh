#!/bin/bash
set -euxo pipefail
apt-get clean && apt-get -y update
apt-get -y install wget && add-apt-repository contrib && apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update -y && apt-get -y install cuda-toolkit
apt-get install -y zlib1g cudnn9-cuda-12
python3 -m pip install --upgrade pip && python3 -m pip install wheel && python3 -m pip install --upgrade tensorrt

#export CUDA_PATH=/usr/local/cuda
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/cuda-12.6/lib64:/usr/local/lib/python3.10/dist-packages/tensorrt_libs/
#echo "Done: I have exported LD_LIBRARY_PATH correctly."