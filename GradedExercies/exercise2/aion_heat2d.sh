#!/bin/bash -l
#SBATCH --job-name=heat_2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=0-01:00:00
#SBATCH --partition=batch
#SBATCH --reservation=mpi_course
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "== Starting job ${SLURM_JOBID} at $(date)"

serial_exe="serial_2d_heat.out"
mpi_exe="mpi_2d_heat.out"

module load toolchain/foss
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

outdir="results_${SLURM_JOB_ID}"
mkdir -p "$outdir"

echo "Running serial heat2d baseline"
./"$serial_exe" > "$outdir/out_serial.txt"

for n in 1 2 4 8 16 32 64 128; do
    echo "Running heat2d with $n MPI ranks"

    srun --exact --ntasks="$n" --cpus-per-task=1 \
         --cpu-bind=cores "$mpi_exe" > "$outdir/out_${n}.txt"
done

echo "All runs complete. Output: $outdir/"
echo "== Finished job ${SLURM_JOBID} at $(date)"
