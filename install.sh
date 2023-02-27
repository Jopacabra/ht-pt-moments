#!/bin/bash

# go to project root (directory above this script)
# https://stackoverflow.com/a/246128
# cd "$(dirname "${BASH_SOURCE[0]}")"/..
# ALREADY IN PROPER PATH

# Create python virtual environment
export pkgname='jma'
export VIRTUAL_ENV=$pkgname

# create python virtual environment to install into
python3 -m venv $VIRTUAL_ENV
# Activate environment
source $VIRTUAL_ENV/bin/activate
# Install python dependencies - Excludes tkinter for plasma inspector
pip install numpy scipy cython h5py pandas xarray matplotlib pyyaml

# Install freestream - required before installing osu-hydro
# subshell allows temporary environment modification
cd freestream
(
  [[ $PY_FLAGS ]] && export CFLAGS=$PY_FLAGS CXXFLAGS=$PY_FLAGS
  exec python3 setup.py install
) || exit 1
cd ..

# Install frzout - required before installing osu-hydro
cd frzout
(
  [[ $PY_FLAGS ]] && export CFLAGS=$PY_FLAGS CXXFLAGS=$PY_FLAGS
  exec python3 setup.py install
) || exit 1
cd ..

# Build trento
cd trento
# Remove the build, if present
if [[ -d build ]]; then
      rm -rf build
fi
# Create and enter build directory
mkdir build && cd build
# Generate cmake business
cmake3 ..
# Install the module
make install
cd ..
cd ..

# Build osu-hydro
cd osu-hydro
# Remove the build, if present
if [[ -d build ]]; then
      rm -rf build
fi
# Create and enter build directory
mkdir build && cd build
# Generate cmake business
cmake3 .. 
# Install the module
make install
cd ..
cd ..

# install the event runner script into bin
# install -v EBE.py "$prefix/bin"
