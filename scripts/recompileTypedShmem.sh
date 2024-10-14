#!/bin/bash

# Script to compile a wheel and install it to the current conda environment

# Exit immediately if a command exits with a non-zero status
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if conda is activated
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Error: No conda environment is currently activated."
    echo "Please activate a conda environment and try again."
    exit 1
fi

# Check if necessary commands exist
for cmd in python3 pip conda; do
    if ! command_exists $cmd; then
        echo "Error: $cmd is not installed or not in PATH"
        exit 1
    fi
done

# Store current directory
curDir="$PWD"

# Change to the TypedShmem directory
cd wistex-system/Util/TypedShmem || { echo "Error: Directory not found"; exit 1; }

echo "Building wheel..."
python3 -m build -w --verbose --no-isolation || { echo "Error: Wheel build failed"; exit 1; }

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')

# Get OS
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

# Get architecture
ARCH=$(uname -m)

# Construct wheel filename
WHEEL_FILE="typedshmem-0.0.1-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-${OS}_${ARCH}.whl"

echo "Installing wheel: $WHEEL_FILE"
if [ -f "dist/$WHEEL_FILE" ]; then
    pip install "dist/$WHEEL_FILE" --no-clean --force-reinstall || { echo "Error: Wheel installation failed"; exit 1; }
    # ps: --no-clean keep the debug symbols, which makes debugging much easier
else
    echo "Error: Wheel file not found: $WHEEL_FILE"
    echo "Available wheel files:"
    ls dist/*.whl
    exit 1
fi

# Return to original directory
cd "$curDir" || { echo "Error: Unable to return to original directory"; exit 1; }

echo "Wheel compiled and installed successfully!"