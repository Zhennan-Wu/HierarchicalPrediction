#!/bin/bash

#SBATCH -J dbm
#SBATCH -p gpu
#SBATCH -o ../outputs/output_%j.txt
#SBATCH -e ../outputs/error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

#Load any modules that your program needs
module load conda/24.1.2
module load nvidia/21.5

conda activate hdp

#Run your program
srun python3 ./dbn.py 
srun python3 ./dbm.py