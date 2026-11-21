#!/bin/bash -l
#SBATCH --job-name mpi_abc
#SBATCH -N 1
#SBATCH --ntasks-per-node 128 # MPI processes per node
#SBATCH --cpus-per-task 1 # Cores per MPI process (OpenMP)
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --reservation=mpi_course 
#SBATCH --exclusive 
#SBATCH --output slurm_%x_%j.log
#SBATCH --error  slurm_%x_%j.log

echo "== Starting job ${SLURM_JOBID} at $(date)"

module load toolchain/foss
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for i in 1 2 4 8 16 32 64 128 ; do
    srun -n $i mpi_abc_split_ab.out
done

echo "== Finished job ${SLURM_JOBID} at $(date)"