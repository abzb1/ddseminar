#!/bin/bash                 
#SBATCH --nodes=1           
#SBATCH --partition=gpu2    
#SBATCH --gres=gpu:a10:1   
#SBATCH --cpus-per-task=14  
#SBATCH -o ./_out/%j.%N.out 
#SBATCH -e ./_err/%j.%N.err 
#================================================
 
echo "start at:" `date`     
echo "node: $HOSTNAME"      
echo "jobid: $SLURM_JOB_ID" 

nvidia-smi                  

# (1) print your name
# (2) excute 01_exam.py python file
# python 01_exam.py
lsb_release -a