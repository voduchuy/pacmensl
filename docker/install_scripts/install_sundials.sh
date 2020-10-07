#!/bin/bash
user=${USERNAME}

sundials_link="https://computing.llnl.gov/projects/sundials/download/sundials-5.2.0.tar.gz"

source /home/${user}/.bashrc
echo ${PETSC_DIR}

cd /home/${user}/software/src
wget ${sundials_link} -O sundials.tar.gz
tar -xf sundials.tar.gz
rm sundials.tar.gz
mv sundials* sundials

cd /home/${user}/software/build
mkdir sundials
cd sundials

echo ${PETSC_DIR}

cmake -DCMAKE_INSTALL_PREFIX=/home/${user}/software/install -DPETSC_ENABLE=ON -DMPI_ENABLE=ON -DPETSC_LIBRARIES=${PETSC_DIR}/lib/libpetsc.so -DPETSC_INCLUDES=${PETSC_DIR}/include \
-DSUNDIALS_INDEX_SIZE=32 /home/${user}/software/src/sundials
make -j4
make install
