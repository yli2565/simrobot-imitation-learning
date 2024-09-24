#!/bin/bash
# This script is pecific to lab machine, which is Ubuntu22.04
# The reason we need this is that the c++ lib comes with conda env is not compatible with the system lib used by python
# The fix is to copy and link the system lib in the conda env

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if we're in a conda environment
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: This script must be run in a conda environment."
    exit 1
fi

# Check if we're on Ubuntu 22.04
if ! grep -q "22.04.4 LTS (Jammy Jellyfish)" /etc/os-release; then
    echo "Error: This script is specific to Ubuntu 22.04."
    exit 1
fi


curDir="$PWD"

cd "$CONDA_PREFIX/lib"

mkdir backup
mv libstd* backup
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./
ln -s libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6 libstdc++.so.6.0.29

cd "$curDir"
