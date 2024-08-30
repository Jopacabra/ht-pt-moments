#!/bin/bash
# Go to home
cd ~

# Update apt-get stuff
sudo -y apt-get update

# Install dependencies
sudo apt-get -y install build-essential libtool autoconf unzip wget
sudo apt-get -y install cmake
sudo apt-get -y install libboost-all-dev
sudo apt-get -y install libhdf5-serial-dev

# Get Conda
exit
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh