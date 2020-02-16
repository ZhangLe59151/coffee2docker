FROM ubuntu:16.04
MAINTAINER ZhangLe<zhanglenus@gmail.com>
LABEL Description="add python3 from ubuntu:16.04" Version="1.0"
RUN su
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y python3
RUN apt-get install -y --no-install-recommends build-essential \
  git \
  libgoogle-glog-dev \
  libgtest-dev \
  libiomp-dev \
  libleveldb-dev \
  liblmdb-dev \
  libopencv-dev \
  libopenmpi-dev \
  libsnappy-dev \
  libprotobuf-dev \
  openmpi-bin \
  openmpi-doc \
  protobuf-compiler \
  python-dev \
  python3-pip \
  python-pip
RUN pip3 install --upgrade pip    
RUN pip install --upgrade pip                  
RUN pip3 install numpy \
  opencv-python \
  scikit-image
RUN pip3 install jupyter
RUN pip3 install matplotlib
RUN pip3 install tensorflow
RUN pip3 install keras
RUN pip3 install imgaug
