# Install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y

# Install required Python packages
pip install -r requirements.txt

# Initialize and update git submodules
git submodule update --init submodules/simple-knn
git submodule update --init submodules/third_party/glm

# delete build and egg-info if exists
if [ -d "submodules/*/build" ]; then
    rm -r submodules/*/build
fi
if [ -d "submodules/*/*.egg-info" ]; then
    rm -r submodules/*/*.egg-info
fi

# Install submodules in editable mode
pip install -e submodules/simple-knn
pip install -e submodules/base-rasterization
pip install -e submodules/full-rasterization
pip install -e submodules/hist-rasterization
pip install -e submodules/gcnt-rasterization

# Install current package in development mode
python setup.py develop
