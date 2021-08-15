FROM ubuntu:latest as builder
MAINTAINER Huy Duc Vo

ARG pacmensl_version=0.1.0

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && \
    apt-get -y install sudo && \
    apt-get -y install dialog apt-utils && \
    apt-get clean

# set environment variables
ENV USERNAME user
ENV PACMENSL_VERSION $pacmensl_version

RUN adduser --disabled-password --gecos --create-home ${USERNAME} && \
    adduser ${USERNAME} sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

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
  make \
"

RUN sudo apt-get install -y ${PACKAGES} && sudo apt-get clean

RUN sudo echo "export LD_LIBRARY_PATH=/usr/local/lib" >> /home/${USERNAME}/.bashrc && \
    mkdir /home/${USERNAME}/software && \
    mkdir /home/${USERNAME}/software/src && \
    mkdir /home/${USERNAME}/software/build

# Copy install scripts
COPY install_scripts/*.sh ./

ENV CPATH=/home/${USERNAME}/software/install/include:${CPATH}
ENV LIBRARY_PATH=/home/${USERNAME}/software/install/lib:${LIBRARY_PATH}
ENV PATH /home/${USERNAME}/anaconda/bin:${PATH}
ENV PETSC_DIR=/home/${USERNAME}/software/install/petsc
ENV PETSC_ARCH=linux-c-opt

RUN sudo chmod +x ./install*.sh &&  \
    # install open-mpi 4.0
    ./install_mpi.sh &&  \
     # install CMake 3.20
    ./install_cmake.sh && \
    # install metis
    ./install_metis.sh && \
    # install Zoltan
    ./install_zoltan.sh && \
    # install armadillo
    ./install_arma.sh

# install anaconda
RUN sudo chmod +x ./install_conda.sh && ./install_conda.sh  \
    && \
    conda update conda && \
    conda init bash

# install petsc
RUN ./install_petsc.sh && \
    ## install sundials
    ./install_sundials.sh

# install pacmensl
RUN sudo chmod +x ./install_pacmensl.sh && ./install_pacmensl.sh

# Cleanup
RUN sudo rm *.sh && \
    sudo rm -rf /home/${USERNAME}/software/build && \
    sudo rm -rf /home/${USERNAME}/software/src

FROM ubuntu:latest

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && \
    apt-get -y install sudo && \
    apt-get clean

# set environment variables
ENV USERNAME user

RUN adduser --disabled-password --gecos --create-home ${USERNAME} && \
    adduser ${USERNAME} sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/${USERNAME}
USER ${USERNAME}
RUN chmod a+rwx /home/${USERNAME}/

# install GCC and other basic libraries
ENV PACKAGES="\
  libopenblas-base \
  openmpi-bin \
  openmpi-common \
"

RUN sudo apt update && \
    sudo apt-get install -y ${PACKAGES} && \
    sudo apt-get clean

# Set OMPI environment variable to prevent spurious printing, see issue 4948 openmpi github page
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
# We're using MPI so don't let OpenBLAS use multi-threading
ENV OMP_NUM_THREADS=1

COPY --from=builder /home/${USERNAME}/software/install/include/* /usr/local/include/
COPY --from=builder /home/${USERNAME}/software/install/lib/* /usr/local/lib/
COPY --from=builder /home/${USERNAME}/software/install/petsc/include/* /usr/local/include/
COPY --from=builder /home/${USERNAME}/software/install/petsc/lib/* /usr/local/lib/
COPY --from=builder /home/${USERNAME}/software/install/examples .

ENV LIBRARY_PATH=/usr/local/lib:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
