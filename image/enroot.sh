CONTAINER_PATH="/scratch/enroot/$UID/data/megatron-latest"
CONTAINER_NAME="megatron-latest"
CONTAINER_IMAGE_PATH="$HOME/ddseminar/image/megatron-latest.sqsh"
RELOAD_CONTAINER=true

if $RELOAD_CONTAINER ; then
    rm -rf $CONTAINER_PATH
fi

if [ -d "$CONTAINER_PATH" ] ; then 
    echo "container exist";
else
    enroot create -n $CONTAINER_NAME $CONTAINER_IMAGE_PATH ;
fi

hostnode=`hostname -s`
/usr/local/bin/gpustat -i > $HOME/ddseminar/_log/$hostnode.gpu &

enroot start --root \
            --rw \
            -m $HOME/ddseminar:/ddseminar \
            $CONTAINER_NAME \
            bash