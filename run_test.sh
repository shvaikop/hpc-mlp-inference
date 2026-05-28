#!/bin/bash
#SBATCH -A edu26.dd2356             # Your KTH project account
#SBATCH -J mlp_infer                # Job name
#SBATCH -p main                     # CPU partition on Dardel
#SBATCH -N 1                        # Request 1 node
#SBATCH --ntasks=16                  # 1 MPI rank
#SBATCH --cpus-per-task=8          # 128 cores per rank (1x128=128)
#SBATCH -t 00:10:00                 # Time limit (10 minutes)

# Swap from the default Cray environment to the GNU environment
module swap PrgEnv-cray PrgEnv-gnu

# Set the number of OpenMP threads to match the requested CPUs per task
export OMP_NUM_THREADS=8

# Recommended OpenMP thread binding settings on Dardel to prevent core contention
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# WARNING: Ensure there are NO trailing spaces after the backslashes (\) below
# Added --cpu-bind=cores so Slurm maps the MPI ranks correctly to the physical cores
srun --cpu-bind=cores ./build/mlp_infer_mpi \
  --export-dir cifar10_embedding_mlp/export \
  --input cifar10_embedding_mlp/export/test_input_embeddings_all.bin \
  --batch-size 128 \
  --quiet