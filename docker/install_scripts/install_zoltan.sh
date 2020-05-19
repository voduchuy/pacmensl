#!/bin/bash

cd /home/user/software/src
git clone https://github.com/trilinos/Trilinos.git
cd ../build
mkdir zoltan
cd zoltan
cmake \
-DTPL_ENABLE_MPI=ON \
-DTrilinos_ENABLE_Zoltan=ON \
-DCMAKE_INSTALL_PREFIX=/usr/local/ \
-DBUILD_SHARED_LIBS=ON \
-DTPL_ENABLE_ParMETIS=ON \
-DParMETIS_INCLUDE_DIRS=/usr/local/include \
-DTrilinos_GENERATE_REPO_VERSION_FILE=OFF \
-DParMETIS_LIBRARY_DIRS=/usr/local/lib \
../../src/Trilinos
make -j4
sudo make install

# clean up
cd /home/user
rm -rf software/src/*
rm -rf software/build/*
