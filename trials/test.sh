#! /bin/bash

#SBATCH --partition Lewis  # use the Lewis partition
#SBATCH --job-name=test-run  # give the job a custom name
#SBATCH --output=test-run-results-%j.out  # give the job output a custom name
#SBATCH --time 0-00:20  # two hour time limit

#SBATCH --nodes=1  # number of nodes
#SBATCH --ntasks=3  # number of cores (AKA tasks)
#SBATCH --mem-per-cpu=1G
# Commands here run only on the first core

# Commands with srun will run on all cores in the allocation
module load openmpi/openmpi-2.0.1
module load python/python-3.5.2
module load mpich/mpich-3.2
module load gurobi/gurobi-8.1.0

mpiexec -n 3 python3 /home/jcb7d7/data/orca-solver/parallel-solver.py /home/jcb7d7/data/orca-solver/inputs/twentyFive.txt 140 141 5 15
