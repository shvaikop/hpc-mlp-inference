#!/bin/bash
#SBATCH -A edu26.dd2356
#SBATCH -J mlp_strong_scaling
#SBATCH -p main
#SBATCH -N 128                    # 改这里: 1, 2, 4, 8
#SBATCH --ntasks-per-node=1     # 每节点 1 个 MPI rank
#SBATCH --cpus-per-task=64     # 每 rank 用满整个节点 (128核)
#SBATCH -t 00:01:00

export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=close

cd /cfs/klemming/home/s/shvaiko/hpc-mlp-inference

srun --cpu-bind=cores ./build/mlp_infer_mpi \
  --export-dir cifar10_embedding_mlp/export \
  --input cifar10_embedding_mlp/export/test_input_embeddings_all.bin \
  --batch-size 128 \
  --num-batches 64 \
  --quiet
