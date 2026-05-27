#!/bin/bash
# Weak scaling on a single node (no SLURM)
# Each rank processes a fixed 8 batches
# OMP threads fixed at 8 per rank
#
# N=1  -> 8  batches total (1024 samples)
# N=2  -> 16 batches total (2048 samples)
# N=4  -> 32 batches total (4096 samples)
# N=8  -> 64 batches total (8192 samples)

BATCHES_PER_RANK=8
BATCH_SIZE=128
OMP=8
EXPORT_DIR=cifar10_embedding_mlp/export
INPUT=cifar10_embedding_mlp/export/test_input_embeddings_all.bin
OUTDIR=benchmark_CPU_Weak

mkdir -p $OUTDIR

for RANKS in 1 2 4 8; do
    NUM_BATCHES=$((RANKS * BATCHES_PER_RANK))
    echo "Running: $RANKS ranks, $NUM_BATCHES total batches..."

    OMP_NUM_THREADS=$OMP \
    OMP_PLACES=cores \
    OMP_PROC_BIND=close \
    mpirun -n $RANKS \
        ./build_cpu/mlp_infer_mpi \
        --export-dir $EXPORT_DIR \
        --input $INPUT \
        --batch-size $BATCH_SIZE \
        --num-batches $NUM_BATCHES \
        --quiet \
        > ${OUTDIR}/${RANKS}_${OMP}.out 2>&1

    echo "  Done. Output: ${OUTDIR}/${RANKS}_${OMP}.out"
done

echo "All weak scaling runs complete."
