#!/bin/bash
user=${USERNAME}

cd /home/${user}/software/src
wget http://sourceforge.net/projects/arma/files/armadillo-9.880.1.tar.xz -O arma.tar.xz
tar -xvf arma.tar.xz
mv armadillo-9.880.1 arma

rm *.xz
cd ../build
mkdir arma
cd arma
cmake -DCMAKE_INSTALL_PREFIX=/home/${user}/software/install ../../src/arma
make -j4
make install


