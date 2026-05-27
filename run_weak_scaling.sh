#!/bin/bash
#SBATCH -A edu26.dd2356
#SBATCH -J mlp_weak_scaling
#SBATCH -p main
#SBATCH -N 4                    # 改这里: 1, 2, 4, 8, 16
#SBATCH --ntasks-per-node=1     # 每节点 1 个 MPI rank
#SBATCH --cpus-per-task=64      # 每 rank 64 个 OpenMP 线程
#SBATCH -t 00:10:00

export OMP_NUM_THREADS=64
export OMP_PLACES=cores
export OMP_PROC_BIND=close

cd /cfs/klemming/home/s/shvaiko/hpc-mlp-inference

# Weak scaling: 每个 rank 固定处理 16 batches
# ranks × 16 = --num-batches
# N=1  → --num-batches 16
# N=2  → --num-batches 32
# N=4  → --num-batches 64
# N=8  → --num-batches 128
# N=16 → --num-batches 256

RANKS=$SLURM_NTASKS
NUM_BATCHES=$((RANKS * 16))

srun --cpu-bind=cores ./build/mlp_infer_mpi \
  --export-dir cifar10_embedding_mlp/export \
  --input cifar10_embedding_mlp/export/test_input_embeddings_all.bin \
  --batch-size 128 \
  --num-batches ${NUM_BATCHES} \
  --quiet
