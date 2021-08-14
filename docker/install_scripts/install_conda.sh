#!/bin/bash
user=${USERNAME}

arch=$(uname -m)

if [ "$arch" = "arm64" ] || [ "$arch" = "aarch64" ]; then
  conda_link="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
else
  conda_link="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
fi

cd /home/${user}/software/src
wget ${conda_link} -O conda.sh

sudo chmod u+x ./conda.sh
bash ./conda.sh -b -p /home/${user}/anaconda
rm ./conda.sh
