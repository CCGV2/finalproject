#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=4                # uses 1 compute core per task
#SBATCH --time=3:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=final2-2
#SBATCH --output=final2-2.out
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2020.07
# module load python/intel/3.7.10
source activate /scratch/cg3946/penv
# g++ -I /share/apps/intel/19.1.2/mkl/include/ -L /share/apps/intel/19.1.2/mkl/lib/intel64/ -o dp3 dp3.c -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
# ./dp3
python ./bigtransfer3.py