#!/bin/sh

sudo apt-get install -y libopenmpi-dev openmpi-bin openmpi-common

#user=${USERNAME}
#
#cd /home/${user}/software/src
#
#wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz
#tar -xvf openmpi-4.0.3.tar.gz
#cd openmpi-4.0.3
#mkdir build
#cd build
#../configure prefix=/usr/local/openmpi
#make -j4
#sudo make install
#
#echo "export PATH=${PATH}:/usr/local/openmpi/bin" >> ~/.bashrc
#echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/openmpi/lib" >> ~/.bashrc
#echo "export CPATH=${CPATH}:/usr/local/openmpi/include"
#
## clean up
#cd /home/${user}
#rm -rf software/src/openmpi*
