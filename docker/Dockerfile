FROM continuumio/anaconda3:latest
FROM tensorflow/tensorflow:latest-gpu-py3

RUN addgroup --gid 551 eingerman
RUN useradd -r -u 551 -g eingerman eingerman

ARG PYTHON_VERSION=3.6
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libav-tools wget git vim

RUN git clone https://github.com/geneing/Tacotron-2.git
WORKDIR Tacotron-2
RUN pip3 install -r requirements.txt

USER eingerman
WORKDIR /workspace
