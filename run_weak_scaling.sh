#!/bin/bash
#SBATCH -A edu26.dd2356
#SBATCH -J mlp_weak_scaling
#SBATCH -p shared
#SBATCH -N 4                    # Change here: 1, 2, 4, 8
#SBATCH --ntasks-per-node=1     # 1 MPI rank per node
#SBATCH --cpus-per-task=64      # 64 OpenMP threads per rank
#SBATCH -t 00:10:00

export OMP_NUM_THREADS=64
export OMP_PLACES=cores
export OMP_PROC_BIND=close

cd /cfs/klemming/home/s/shvaiko/hpc-mlp-inference

# Weak scaling: each rank processes a fixed 8 batches
# Input file has 10000 samples = 78 full batches (batch_size=128).
# Max safe total = 64 batches (8192 samples) => 8 batches/rank * 8 ranks.
#
# N=1 --> --num-batches 8
# N=2 --> --num-batches 16
# N=4 --> --num-batches 32
# N=8 --> --num-batches 64

RANKS=$SLURM_NTASKS
NUM_BATCHES=$((RANKS * 8))

srun --cpu-bind=cores ./build/mlp_infer_mpi \
  --export-dir cifar10_embedding_mlp/export \
  --input cifar10_embedding_mlp/export/test_input_embeddings_all.bin \
  --batch-size 128 \
  --num-batches ${NUM_BATCHES} \
  --quiet
