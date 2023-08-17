#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=14
#SBATCH -o ./_out/%j.sbatch.%N.out
#SBATCH -e ./_err/%j.sbatch.%N.err
#=============================================================
GRES="gpu:a10:1"                   
. 03_conf.sh
#==============================================================
mkdir -p ./_log/$SLURM_JOB_ID      

echo CONTAINER_PATH:$CONTAINER_PATH

INIT_CONTAINER_SCRIPT=$(cat <<EOF

    if [ -d "$CONTAINER_PATH" ] ; then 
        echo "container exist";
    else
        enroot create -n $CONTAINER_NAME $CONTAINER_IMAGE_PATH ;
    fi

EOF
)

MONITOR_GPU_SCRIPT=$(cat <<EOF
    hostnode=\`hostname -s\`
    /usr/local/bin/gpustat -i > $HOME/ddseminar/slurm/_log/$SLURM_JOB_ID/\$hostnode.gpu &
EOF
)

SRUN_SCRIPT=$(cat <<EOF

    $INIT_CONTAINER_SCRIPT

    $MONITOR_GPU_SCRIPT

    enroot start --root \
                --rw \
                -m $HOME/ddseminar:/ddseminar \
                $CONTAINER_NAME \
                python /ddseminar/slurm/03_train.py
EOF
)

srun --partition=$SLURM_JOB_PARTITION \
      --gres=$GRES \
      --cpus-per-task=14 \
      -o ./_log/%j/%N.out \
      -e ./_log/%j/%N.err \
      bash -c "$SRUN_SCRIPT"
