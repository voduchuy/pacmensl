FROM ubuntu:latest

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && \
    apt-get -y install sudo && \
    apt-get clean
RUN apt-get -y install dialog apt-utils

# set environment variables
ENV USERNAME huy

RUN adduser --disabled-password --gecos --create-home ${USERNAME}
RUN adduser ${USERNAME} sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/${USERNAME}
USER ${USERNAME}
RUN chmod a+rwx /home/${USERNAME}/

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
  xz-utils \
  git \
  python3.8 \
"

RUN sudo apt-get install -y ${PACKAGES} && sudo apt-get clean


RUN sudo echo "export LD_LIBRARY_PATH=/usr/local/lib" >> /home/${USERNAME}/.bashrc && \
    mkdir /home/${USERNAME}/software && \
    mkdir /home/${USERNAME}/software/src && \
    mkdir /home/${USERNAME}/software/build
#
## install open-mpi 4.0
#COPY install_scripts/install_mpi.sh ./
#RUN sudo chmod +x ./install*.sh && ./install_mpi.sh
#
## install CMake 3.20
#COPY install_scripts/install_cmake.sh ./
#RUN sudo chmod +x ./install_cmake.sh && ./install_cmake.sh
#
## install metis
#COPY install_scripts/install_metis.sh ./
#RUN sudo chmod +x ./install_metis.sh && ./install_metis.sh
#
## install Zoltan
#COPY install_scripts/install_zoltan.sh ./
#RUN sudo chmod +x ./install_zoltan.sh && ./install_zoltan.sh
#
## install armadillo
#COPY install_scripts/install_arma.sh ./
#RUN sudo chmod +x ./install_arma.sh && ./install_arma.sh

# install anaconda
COPY install_scripts/install_conda.sh ./
ENV PATH /home/${USERNAME}/anaconda/bin:${PATH}
RUN sudo chmod +x ./install_conda.sh && ./install_conda.sh  \
    && \
    conda update conda && \
    conda init bash


# install petsc
#COPY install_scripts/install_petsc.sh ./
#RUN sudo chmod +x ./install_petsc.sh && ./install_petsc.sh
#

#
## install sundials
#COPY install_scripts/install_sundials.sh ./
#ENV PETSC_DIR=/home/${USERNAME}/software/install/petsc
#ENV PETSC_ARCH=linux-c-opt
#RUN sudo chmod +x ./install_sundials.sh && ./install_sundials.sh
#
## install pacmensl
#COPY install_scripts/install_pacmensl.sh ./
#ENV CPATH=/home/${USERNAME}/software/install/include
#ENV LIBRARY_PATH=/home/${USERNAME}/software/install/lib
#RUN sudo chmod +x ./install_pacmensl.sh && ./install_pacmensl.sh
#
#
