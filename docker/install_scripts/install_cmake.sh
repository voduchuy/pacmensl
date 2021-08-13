user=${USERNAME}

sudo apt-get -y install build-essential libssl-dev

cd /home/${user}/software/src
wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
tar -zxvf cmake-3.20.0.tar.gz
cd cmake-3.20.0
./bootstrap
make
sudo make install