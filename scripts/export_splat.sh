EXP_PATH=$1
EXP_NAME=${EXP_PATH##*/}
ITER=${2:-latest}

if [ ! -f "submodules/hdr-splat/convert.py" ]; then
    echo "Error: submodules/hdr-splat/convert.py not found; updating submodules..."
    git submodule update --init --recursive submodules/hdr-splat
fi

if [ ! -d "output/splat/$EXP_NAME" ]; then
    mkdir -p output/splat/$EXP_NAME
fi

python submodules/hdr-splat/convert.py $EXP_PATH/models/net_g_$ITER.ply -o output/splat/$EXP_NAME/$ITER.splat
cp $EXP_PATH/meta_data.json output/splat/$EXP_NAME/meta_data.json
