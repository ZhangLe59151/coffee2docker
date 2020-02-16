FROM ubuntu:16.04
MAINTAINER ZhangLe <zhanglenus@gmail.com>
LABEL Description="add python3 from ubuntu:16.04" Version="1.0"
RUN su
RUN apt-get update
RUN apt-get install -y python
RUN apt-get install -y --no-install-recommends build-essential \
RUN apt-get install -y git \
RUN apt-get install -y libgoogle-glog-dev \
RUN apt-get install -y libgtest-dev \
RUN apt-get install -y libiomp-dev \
RUN apt-get install -y libleveldb-dev \
RUN apt-get install -y liblmdb-dev \
RUN apt-get install -y libopencv-dev \
RUN apt-get install -y libopenmpi-dev \
RUN apt-get install -y libsnappy-dev \
RUN apt-get install -y libprotobuf-dev \
RUN apt-get install -y openmpi-bin \
RUN apt-get install -y openmpi-doc \
RUN apt-get install -y protobuf-compiler \
RUN apt-get install -y python-dev \
RUN apt-get install -y python-pip \ 
RUN pip install --upgrade pip \                        
#RUN pip install --user \
#      future \
#      numpy \
#      protobuf \
#      typing \
#      hypothesis
# RUN pip install opencv-python \
# RUN sudo apt-get install -y --no-install-recommends \
#      libgflags-dev \
#      cmake