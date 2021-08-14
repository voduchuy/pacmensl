#!/bin/bash
user=${USERNAME}

conda_link="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
cd /home/${user}/software/src
wget ${conda_link} -O conda.sh

sudo chmod u+x ./conda.sh
bash ./conda.sh -b -p /home/${user}/anaconda
rm ./conda.sh
