#!/bin/bash
user=${USERNAME}

cd /home/${user}/software/src
git clone https://github.com/trilinos/Trilinos.git --depth 1 --branch master --single-branch
cd ../build
mkdir zoltan
cd zoltan
cmake \
-DCMAKE_INSTALL_PREFIX=/home/${user}/software/install \
-DTPL_ENABLE_MPI=ON \
-DTrilinos_ENABLE_Zoltan=ON \
-DBUILD_SHARED_LIBS=ON \
-DTPL_ENABLE_ParMETIS=ON \
-DParMETIS_INCLUDE_DIRS=/usr/local/include \
-DTrilinos_GENERATE_REPO_VERSION_FILE=OFF \
-DParMETIS_LIBRARY_DIRS=/home/${user}/software/install/lib \
../../src/Trilinos
make -j4
make install



