# Check for root permission
if [ "$EUID" -ne 0 ]; then
    echo "Please use root permission to run this script"
    exit 1
fi

# Set CUDA installation variables
CUDA_VERSION="11.8.0"
CUDA_INSTALLER="cuda_${CUDA_VERSION}_520.61.05_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_INSTALLER}"

# Create and enter installation directory
INSTALL_DIR="submodules/cuda"
mkdir -p $INSTALL_DIR
pushd $INSTALL_DIR > /dev/null

# Download CUDA installer if not exists
if [ ! -f $CUDA_INSTALLER ]; then
    echo "Downloading CUDA ${CUDA_VERSION}..."
    wget $CUDA_URL
fi

# Install CUDA
echo "Installing CUDA ${CUDA_VERSION}... in silent mode"
sh $CUDA_INSTALLER --silent --toolkit

# Set environment variables
CUDA_PATH="/usr/local/cuda"
CUDA_ENV_VARS=(
    "export PATH=${CUDA_PATH}/bin:\$PATH"
    "export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH"
)

# Choose profile file based on shell type
PROFILE_PATH=$([ -n "$ZSH_VERSION" ] && echo ~/.zshrc || echo ~/.bashrc)

# Add environment variables if not exists
for var in "${CUDA_ENV_VARS[@]}"; do
    if ! grep -q "$var" $PROFILE_PATH; then
        echo "$var" >> $PROFILE_PATH
    fi
done

# Apply environment variables
source $PROFILE_PATH

# Return to original directory
popd > /dev/null
