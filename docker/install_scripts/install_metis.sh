#!/bin/bash

cd /home/user/
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar -xvf metis-5.1.0.tar.gz
cd metis-5.1.0
make config shared=1
make -j4
sudo make install

cd /home/user/software/src
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xvf parmetis-4.0.3.tar.gz
cd parmetis-4.0.3
make config shared=1
make -j4
sudo make install

# clean up source and build files
cd /home/user
rm -rf software/src/*
