FROM ubuntu:16.04
MAINTAINER ZhangLe <zhanglenus@gmail.com>
LABEL Description="add python3 from ubuntu:16.04" Version="1.0"
RUN su
RUN apt-get update
RUN apt-get install -y python
RUN apt-get install -y --no-install-recommends \
      build-essential \
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
      python-pip  
RUN pip install --upgrade pip\                        
RUN pip install --user \
      future \
      numpy \
      protobuf \
      typing \
      hypothesis
RUN pip install opencv-python\
RUN sudo apt-get install -y --no-install-recommends \
      libgflags-dev \
      cmake