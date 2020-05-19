#!/bin/sh

user=${USERNAME}

cd /home/${user}/software/src

wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.13.1.tar.gz -O petsc.tar.gz
tar -xvf petsc.tar.gz
rm petsc.tar.gz
mv petsc* petsc
# git clone -b maint https://gitlab.com/petsc/petsc.git petsc
cd petsc
export PETSC_DIR=`pwd`; unset PETSC_ARCH; ./configure PETSC_ARCH=linux-c-opt --with-precision=double --with-scalar-type=real --with-debugging=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 --with-fc=0 --with-shared-libraries=1 --with-avx512-kernels 1 --prefix=/usr/local/petsc
make -j2 PETSC_DIR=/home/${user}/software/src/petsc PETSC_ARCH=linux-c-opt all
sudo make PETSC_DIR=/home/${user}/software/src/petsc PETSC_ARCH=linux-c-opt install

# clean up
cd /home/${user}
rm -rf software/src/petsc*

# add petsc to environment variables
echo "export PETSC_DIR=/usr/local/petsc" >> /home/${user}/.bashrc
echo "export PETSC_ARCH=linux-c-opt" >> /home/${user}/.bashrc
