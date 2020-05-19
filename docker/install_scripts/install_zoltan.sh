#!/bin/bash
user=${USERNAME}

cd /home/${user}/software/src
git clone https://github.com/trilinos/Trilinos.git --depth 1 --branch master --single-branch
cd ../build
mkdir zoltan
cd zoltan
cmake \
-DTPL_ENABLE_MPI=ON \
-DTrilinos_ENABLE_Zoltan=ON \
-DCMAKE_INSTALL_PREFIX=/usr/local/zoltan \
-DBUILD_SHARED_LIBS=ON \
-DTPL_ENABLE_ParMETIS=ON \
-DParMETIS_INCLUDE_DIRS=/usr/local/include \
-DTrilinos_GENERATE_REPO_VERSION_FILE=OFF \
-DParMETIS_LIBRARY_DIRS=/usr/local/lib \
../../src/Trilinos
make -j4
sudo make install

echo "#Add zoltan library paths to environent variables" >> ~/.bashrc
echo "export PATH=${PATH}:/usr/local/zoltan/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/zoltan/lib" >> ~/.bashrc
echo "export LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/zoltan/lib" >> ~/.bashrc
echo "export CPATH=${CPATH}:/usr/local/zoltan/include" >> ~/.bashrc

# clean up
cd /home/${user}
rm -rf software/src/Trilinos
rm -rf software/build/zoltan
