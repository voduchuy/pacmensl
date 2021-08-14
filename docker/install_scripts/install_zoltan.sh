#!/bin/bash
user=${USERNAME}

trilinos_link=https://github.com/trilinos/Trilinos/archive/refs/tags/trilinos-release-13-0-1.tar.gz

cd /home/${user}/software/src
wget ${trilinos_link} -O trilinos.tar.gz
tar -xf trilinos.tar.gz
rm trilinos.tar.gz
mv Trilinos* Trilinos

cd ../build
mkdir zoltan
cd zoltan
cmake \
-DCMAKE_INSTALL_PREFIX=/home/${user}/software/install \
-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
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

# Cleanup
rm -rf /home/${user}/software/src/Trilinos



