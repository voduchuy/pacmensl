#!/bin/bash
user=${USERNAME}

export CPATH=${CPATH};/home/${user}/software/install/include

cd /home/${user}/software/src
git clone https://github.com/voduchuy/pacmensl pacmensl

cd /home/${user}/software/build
mkdir pacmensl
cd pacmensl

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=/home/${user}/software/install /home/${user}/software/src/pacmensl
make -j4

# Now install
make install

# Cleanup all redundant dependencies
rm -rf /home/${user}/software/src