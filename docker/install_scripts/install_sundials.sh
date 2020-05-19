#!/bin/bash

sundials_link="https://computing.llnl.gov/projects/sundials/download/sundials-5.2.0.tar.gz"

cd /home/software/src
wget ${sundials_link} -O sundials.tar.gz
tar -xvf sundials.tar.gz
rm sundials.tar.gz
mv sundials* sundials
cd /home/user/software
mkdir build
cd /home/user/software/build
mkdir sundials
cd sundials
cmake -DPETSC_ENABLE=ON -DMPI_ENABLE=ON -DPETSC_INCLUDE_DIR=/usr/local/petsc/include -DPETSC_LIBRARY_DIR=/usr/local/petsc/lib /home/software/src/sundials
