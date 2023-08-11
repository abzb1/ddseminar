#!/bin/bash
#SBATCH --nodes=2
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=14
#SBATCH -o ./_out/%j.sbatch.%N.out
#SBATCH -e ./_err/%j.sbatch.%N.err
#================================================
GRES="gpu:a10:1"
mkdir -p ./_log/$SLURM_JOB_ID

# print sbatch job 
echo "start at:" `date`
echo "node: $HOSTNAME"
echo "jobid: $SLURM_JOB_ID"

srun --partition=$SLURM_JOB_PARTITION \
     --gres=$GRES \
     --cpus-per-task=14 \
     -o ./_log/%j/%N.out \
     -e ./_log/%j/%N.err \
     python 01_exam.py