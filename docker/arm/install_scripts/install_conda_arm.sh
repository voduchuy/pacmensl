#!/bin/bash
user=${USERNAME}

conda_link="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
cd /home/${user}/software/src
wget ${conda_link} -O conda.sh

sudo chmod u+x ./conda.sh
bash ./conda.sh -b -p /home/${user}/anaconda
rm ./conda.sh
