#!/bin/bash

# conda_link="https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh"
# cd /home/user/software/src
# wget ${conda_link} -O conda.sh

sudo chmod u+x ./conda.sh
bash ./conda.sh -b -p /home/user/anaconda
rm ./conda.sh
