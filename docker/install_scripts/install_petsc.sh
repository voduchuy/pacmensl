#!/bin/bash

user=${USERNAME}

cd /home/${user}/software/src

wget https://gitlab.com/petsc/petsc/-/archive/v3.13.6/petsc-v3.13.6.tar.gz -O petsc.tar.gz
tar -xvf petsc.tar.gz
rm petsc.tar.gz
mv petsc* petsc

cd petsc

arch=$(uname -m)

if [ "$arch" = "arm64" ] || [ "$arch" = "aarch64" ]; then
  export PETSC_DIR=`pwd`; unset PETSC_ARCH; ./configure PETSC_ARCH=linux-c-opt --with-precision=double --with-scalar-type=real --with-debugging=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 --with-fc=0 --with-shared-libraries=1 --prefix=/home/${user}/software/install/petsc
else
  export PETSC_DIR=`pwd`; unset PETSC_ARCH; ./configure PETSC_ARCH=linux-c-opt --with-precision=double --with-scalar-type=real --with-debugging=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 --with-fc=0 --with-shared-libraries=1 --with-avx512-kernels 1 --prefix=/home/${user}/software/install/petsc
fi

make -j4 PETSC_DIR=/home/${user}/software/src/petsc PETSC_ARCH=linux-c-opt all
make PETSC_DIR=/home/${user}/software/src/petsc PETSC_ARCH=linux-c-opt install

# add petsc to environment variables
echo "export PETSC_DIR=/home/${user}/software/install/petsc" >> /home/${user}/.bashrc
echo "export PETSC_ARCH=linux-c-opt" >> /home/${user}/.bashrc
echo "export LD_LIBRARY_PATH=/home/${user}/software/install/petsc/lib:/usr/local/lib:/home/${user}/software/install/lib:${LD_LIBRARY_PATH}" >> /home/${user}/.bashrc
echo "export LIBRARY_PATH=home/${user}/software/install/petsc/lib:/usr/local/lib:/home/${user}/software/install/lib:${LIBRARY_PATH}" >> /home/${user}/.bashrc

# cleanup
rm -rf /home/${user}/software/src/petsc
