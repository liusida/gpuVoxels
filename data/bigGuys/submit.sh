#!/bin/bash
# specify a partition
#SBATCH --partition=dg-jup
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=32
# Request GPUs
#SBATCH --gres=gpu:8
# Request memory 
#SBATCH --mem=512G
# Maximum runtime of 10 minutes
#SBATCH --time=2:00:00
# Name of this job
#SBATCH --job-name=BigGuy
# Output of this job, stderr and stdout are joined by default
# %x=job-name %j=jobid
#SBATCH --output=stdout_%x_%j.out
# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}
time ./Voxelyze3 -i . -o output.xml -lf > 50x_good.history