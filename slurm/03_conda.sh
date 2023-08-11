#!/bin/bash                       
#SBATCH --nodes=1                  
#SBATCH --partition=gpu2          
#SBATCH --gres=gpu:a10:2          
#SBATCH --cpus-per-task=14         
#SBATCH -o ./_out/%j.sbatch.%N.out 
#SBATCH -e ./_err/%j.sbatch.%N.err 
#================================================
GRES="gpu:a10:2"                  
mkdir -p ./_log/$SLURM_JOB_ID     

module purge                      
module add CUDA/11.3.0             
module add ANACONDA/2021.05        

srun --partition=$SLURM_JOB_PARTITION \
     --gres=$GRES \
     --cpus-per-task=14 \
     -o ./_log/%j/%N.out \
     -e ./_log/%j/%N.err \
     $HOME/miniconda3/envs/myenv/bin/python 03_conda.py