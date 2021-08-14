#!/bin/bash
user=${USERNAME}

cd /home/${user}/software/src
git clone https://github.com/voduchuy/pacmensl pacmensl

cd /home/${user}/software/build
mkdir pacmensl
cd pacmensl

cmake -DCMAKE_INSTALL_PREFIX=/home/${user}/software/install /home/${user}/software/src/pacmensl
make -j4
make install
