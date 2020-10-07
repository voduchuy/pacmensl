#!/bin/sh

user=${USERNAME}

cd /home/${user}/software/src

wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.13.1.tar.gz -O petsc.tar.gz
tar -xvf petsc.tar.gz
rm petsc.tar.gz
mv petsc* petsc
# git clone -b maint https://gitlab.com/petsc/petsc.git petsc
cd petsc
export PETSC_DIR=`pwd`; unset PETSC_ARCH; ./configure PETSC_ARCH=linux-c-opt --with-precision=double --with-scalar-type=real --with-debugging=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 --with-fc=0 --with-shared-libraries=1 --with-avx512-kernels 1 --prefix=/home/${user}/software/install/petsc
make -j4 PETSC_DIR=/home/${user}/software/src/petsc PETSC_ARCH=linux-c-opt all
make PETSC_DIR=/home/${user}/software/src/petsc PETSC_ARCH=linux-c-opt install

# add petsc to environment variables
echo "export PETSC_DIR=/home/${user}/software/install/petsc" >> /home/${user}/.bashrc
echo "export PETSC_ARCH=linux-c-opt" >> /home/${user}/.bashrc
echo "export LD_LIBRARY_PATH=home/${user}/software/install/petsc/lib:${LD_LIBRARY_PATH}" >> /home/${user}/.bashrc
echo "export LIBRARY_PATH=home/${user}/software/install/petsc/lib:${LIBRARY_PATH}" >> /home/${user}/.bashrc


