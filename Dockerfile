FROM ubuntu:16.04
MAINTAINER ZhangLe <zhanglenus@gmail.com>
LABEL Description="add python3 from ubuntu:16.04" Version="1.0"
RUN su
RUN apt-get update
RUN apt-get install -y python