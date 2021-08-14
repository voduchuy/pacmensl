#!/bin/bash
user=${USERNAME}

cd /home/${user}/software/src
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar -xvf metis-5.1.0.tar.gz
cd metis-5.1.0
make config prefix=/home/${user}/software/install shared=1
make -j4
make install

cd /home/${user}/software/src
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xvf parmetis-4.0.3.tar.gz
cd parmetis-4.0.3
make config prefix=/home/${user}/software/install shared=1
make -j4
make install

# Cleanup
rm -rf /home/${user}/software/src/*metis*



