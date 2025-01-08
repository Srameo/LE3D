#!/bin/bash

# Print usage information and available options
function usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i|--interactive             Interactive installation"
    echo "  -cuda|--install-cuda            Install CUDA"
    echo "  -colmap|--install-colmap          Install COLMAP"
    echo "      cuda_enabled          Enable CUDA support (must follow --install-colmap)"
    echo "  -env|--create-env              Create and activate conda environment"
    echo "By default, only Python packages will be installed"
}

# Print current installation status and settings
function status() {
    echo "----------------------------------------"
    echo "INSTALL_CUDA: $INSTALL_CUDA"
    echo "INSTALL_COLMAP: $INSTALL_COLMAP"
    echo "  COLMAP_CUDA_ENABLED: $COLMAP_CUDA_ENABLED"
    echo "CREATE_ENV: $CREATE_ENV"
    echo "INTERACTIVE: $INTERACTIVE"
    echo "INSTALL_PYTHON_PACKAGES: $INSTALL_PYTHON_PACKAGES"
    echo "----------------------------------------"
}

# Parse command line arguments
function parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -cuda|--install-cuda)
                INSTALL_CUDA=true
                shift
                ;;
            -colmap|--install-colmap)
                INSTALL_COLMAP=true
                shift
                if [[ "$1" == "cuda_enabled" ]]; then
                    COLMAP_CUDA_ENABLED=true
                    shift
                fi
                ;;
            -env|--create-env)
                CREATE_ENV=true
                shift
                ;;
            -i|--interactive)
                INTERACTIVE=true
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
}

# Interactive setup to configure installation options
function interactive_setup() {
    read -p "Do you want to install CUDA? Root permission required. (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        INSTALL_CUDA=true
    fi

    read -p "Do you want to install COLMAP? Root permission required. (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        INSTALL_COLMAP=true
        read -p "Do you want to enable CUDA support? (y/N): " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            COLMAP_CUDA_ENABLED=true
        fi
    fi

    read -p "Do you want to create and activate a conda environment called 'basicgs'? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        CREATE_ENV=true
    fi
}

# Install CUDA if requested
function install_cuda() {
    if [ "$INSTALL_CUDA" = true ]; then
        echo "Installing CUDA... by \`sudo bash scripts/install/cuda.sh\`"
        sudo bash scripts/install/cuda.sh

	# activate CUDA extension
	export PATH=/usr/local/cuda/bin:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    fi
}

# Install COLMAP with or without CUDA support
function install_colmap() {
    if [ "$INSTALL_COLMAP" = true ]; then
        echo "Installing COLMAP... by \`sudo bash scripts/install/colmap.sh\`"
        if [ "$COLMAP_CUDA_ENABLED" = true ]; then
            sudo bash scripts/install/colmap.sh --cuda-enabled
        else
            sudo bash scripts/install/colmap.sh
        fi
    fi
}

# Create and activate conda environment
function setup_conda_env() {
    if [ "$CREATE_ENV" = true ]; then
        echo "Creating and activating conda environment basicgs..."
        eval "$(conda shell.bash hook)"
        conda create -y -n basicgs python=3.10
        conda activate basicgs
    fi
}

# Install required Python packages
function install_python_packages() {
    if [ "$INSTALL_PYTHON_PACKAGES" = true ]; then
        echo "Installing Python packages... by \`bash scripts/install/python_packages.sh\`"
        bash scripts/install/python_packages.sh
    fi
}

# Initialize default values for installation options
INSTALL_CUDA=false
INSTALL_COLMAP=false
COLMAP_CUDA_ENABLED=false
CREATE_ENV=false
INTERACTIVE=false
INSTALL_PYTHON_PACKAGES=true  # Install python packages by default

# Main execution flow
parse_args "$@"

if [ "$INTERACTIVE" = true ]; then
    interactive_setup
else
    usage
fi

status
install_cuda
install_colmap
setup_conda_env
install_python_packages
