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
cmake ../../src/arma
make -j4
sudo make install

# clean up
cd /home/${user}
rm -rf software/src/arma
rm -rf software/build/arma
