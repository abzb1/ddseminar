#!/bin/bash
#SBATCH --nodes=2
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=56
#SBATCH -o ./_out/%j.sbatch.%N.out
#SBATCH -e ./_err/%j.sbatch.%N.err
#=============================================================
GRES="gpu:a10:4"                 
. conf.sh
#==============================================================
mkdir -p ./_log/$SLURM_JOB_ID

function get_master_adress(){
    NODE_LIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
    MASTER_HOST=`echo $NODE_LIST | awk '{print $1}'`
    MASTER_ADDR=`cat /etc/hosts | grep $MASTER_HOST | awk '{print $1}'`
}
get_master_adress
echo MASTER_ADDR:$MASTER_ADDR
echo CONTAINER_PATH:$CONTAINER_PATH

INIT_CONTAINER_SCRIPT=$(cat <<EOF

    if $RELOAD_CONTAINER ; then
        rm -rf $CONTAINER_PATH
    fi

    if [ -d "$CONTAINER_PATH" ] ; then 
        echo "container exist";
    else
        enroot create -n $CONTAINER_NAME $CONTAINER_IMAGE_PATH ;
    fi

EOF
)

MONITOR_GPU_SCRIPT=$(cat <<EOF
    hostnode=\`hostname -s\`
    /usr/local/bin/gpustat -i > $HOME/ddseminar/grover_ddp/_log/$SLURM_JOB_ID/\$hostnode.gpu &
EOF
)

ENROOT_SCRIPT=" cd /ddseminar/grover_ddp && \
                bash sbatch_torchrun.sh"

SRUN_SCRIPT=$(cat <<EOF

    $INIT_CONTAINER_SCRIPT

    $MONITOR_GPU_SCRIPT

    NODE_LIST=\`scontrol show hostnames \$SLURM_JOB_NODELIST\`
    node_array=(\$NODE_LIST)
    length=\${#node_array[@]}
    hostnode=\`hostname -s\`
    for (( index = 0; index < length ; index++ )); do
        node=\${node_array[\$index]}
        if [ \$node == \$hostnode ]; then
            LOCAL_RANK=\$index
        fi
    done 

    enroot start --root \
                --rw \
                -m $HOME/ddseminar:/ddseminar \
                $CONTAINER_NAME \
                bash -c "$ENROOT_SCRIPT \$LOCAL_RANK $MASTER_ADDR"
EOF
)				

srun --partition=$SLURM_JOB_PARTITION \
      --gres=$GRES \
      --cpus-per-task=56 \
      -o ./_log/%j/%N.out \
      -e ./_log/%j/%N.err \
      bash -c "$SRUN_SCRIPT"
