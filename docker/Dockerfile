FROM ubuntu:latest

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && \
    apt-get -y install sudo && \
    apt-get clean
RUN apt-get -y install dialog apt-utils

RUN adduser --disabled-password --gecos --create-home user
RUN adduser user sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/user
USER user
RUN chmod a+rwx /home/user/

RUN sudo apt update && \
    sudo apt install -y software-properties-common && \
    sudo add-apt-repository -y ppa:deadsnakes/ppa && \
    sudo apt update && \
    sudo apt-get clean

# install GCC and other basic libraries
ENV PACKAGES="\
  gcc \
  g++ \
  gfortran \
  libopenblas-base \
  libopenblas-dev \
  wget \
  cmake \
  xz-utils \
"

RUN sudo apt-get install -y ${PACKAGES} && sudo apt-get clean

# set environment variables
RUN sudo echo "export LD_LIBRARY_PATH=/usr/local/lib" >> /home/user/.bashrc && \
    mkdir /home/user/software && \
    mkdir /home/user/software/src && \
    mkdir /home/user/software/build

COPY install_scripts ./
COPY conda.sh ./
COPY petsc.tar.gz /home/user/software/src/

# install open-mpi 4.0
RUN sudo chmod +x ./install*.sh && ./install_mpi.sh

# install anaconda
RUN ./install_conda.sh
ENV PATH /home/user/anaconda/bin:${PATH}
RUN conda update conda
RUN conda init bash

# install petsc
RUN ./install_petsc.sh

# install sundials
RUN ./install_sundials.sh

# install metis
RUN ./install_metis.sh

# install Zoltan
RUN ./install_zoltan.sh

# install armadillo
RUN ./install_arma.sh
