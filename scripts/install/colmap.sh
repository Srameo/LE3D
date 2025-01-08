#!/bin/bash

function usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i|--interactive    Interactive installation"
    echo "  --cuda-enabled   Enable CUDA support"
}

# Parse command line arguments
INTERACTIVE=false
CUDA_ENABLED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        --cuda-enabled)
            CUDA_ENABLED=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ "$EUID" -ne 0 ]; then
    echo "Please use root permission to run this script"
    exit 1
fi

# Ask for CUDA support in interactive mode
if [ "$INTERACTIVE" = true ] && [ "$CUDA_ENABLED" = false ]; then
    read -p "Do you want to build from source, CUDA enabled? (Y/n) " response
    if [[ "${response,,}" =~ ^(y|Y|)$ ]]; then
        CUDA_ENABLED=true
    fi
fi

if [ "$CUDA_ENABLED" = true ]; then
    # build from source
    # install pre-requisites
    apt-get update && apt-get install -y \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev

    # download colmap
    git submodule update --init submodules/colmap

    # build colmap with cuda support
    pushd submodules/colmap > /dev/null
    mkdir -p build
    pushd build > /dev/null
    cmake .. -GNinja -DGUI_ENABLED=OFF -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    ninja
    ninja install
    popd > /dev/null
    popd > /dev/null
else
    echo "Skipping CUDA support, directly install using apt install."
    apt-get update && apt-get install -y colmap
fi
