#!/bin/bash

# Check that we're in either a conda env or python virtualenv.
# In a conda env if CONDA_PREFIX points to a valid directory and the python
# executable is in CONDA_PREFIX.
if [[ -d "$CONDA_PREFIX" && "$(which python)" -ef "$CONDA_PREFIX/bin/python" ]]; then
  prefix="$CONDA_PREFIX"
else
  # In a virtualenv if file pyvenv.cfg exists one level above the python
  # executable.  (This is better than checking the VIRTUAL_ENV variable, which
  # may not be set.)
  prefix="$(readlink -f "$(dirname "$(which python)")/..")"
  if [[ ! -f "$prefix/pyvenv.cfg" ]]; then
    echo 'must install in an active python virtualenv or conda env'
    exit 1
  fi
fi

# go to project root (directory above this script)
# https://stackoverflow.com/a/246128
# cd "$(dirname "${BASH_SOURCE[0]}")"/..
# ALREADY IN PROPER PATH

# Build trento
cd trento
# Remove the build, if present
if [[ -d build ]]; then
      rm -rf build
# Create and enter build directory
mkdir build && cd build
# Generate cmake business
cmake3 .. 
# Install the module
make install
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

# Install freestream
# subshell allows temporary environment modification
cd freestream
(
  [[ $PY_FLAGS ]] && export CFLAGS=$PY_FLAGS CXXFLAGS=$PY_FLAGS
  exec python setup.py install
) || exit 1
cd ..

# Install frzout
cd frzout
(
  [[ $PY_FLAGS ]] && export CFLAGS=$PY_FLAGS CXXFLAGS=$PY_FLAGS
  exec python setup.py install
) || exit 1
cd ..


# install the event runner script into bin
# install -v EBE.py "$prefix/bin"
