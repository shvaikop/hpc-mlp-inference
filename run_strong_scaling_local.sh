#!/bin/bash
# Strong scaling on a single node (no SLURM)
# Intel Xeon Platinum 8480C: 112 physical cores
# Strategy: pure MPI (OMP=2), increase rank count = increase total cores used
#
# ranks x omp_threads
#  1 rank  x 1 thread =  1 core
#  2 ranks x 1 thread =  2 cores
#  4 ranks x 1 thread =  4 cores
#  8 ranks x 1 thread =  8 cores
# 16 ranks x 1 thread = 16 cores
# 32 ranks x 1 thread = 32 cores
# 64 ranks x 1 thread = 64 cores

TOTAL_BATCHES=64
BATCH_SIZE=128
OMP=2
EXPORT_DIR=cifar10_embedding_mlp/export
INPUT=cifar10_embedding_mlp/export/test_input_embeddings_all.bin
OUTDIR=benchmark_CPU_Strong

mkdir -p $OUTDIR

for RANKS in 1 2 4 8 16 32 64; do
    echo "Running: $RANKS ranks x $OMP OMP thread..."

    OMP_NUM_THREADS=$OMP \
    OMP_PLACES=cores \
    OMP_PROC_BIND=close \
    mpirun -n $RANKS \
        ./build_cpu/mlp_infer_mpi \
        --export-dir $EXPORT_DIR \
        --input $INPUT \
        --batch-size $BATCH_SIZE \
        --num-batches $TOTAL_BATCHES \
        --quiet \
        > ${OUTDIR}/${RANKS}_${OMP}.out 2>&1

    echo "  Done. Output: ${OUTDIR}/${RANKS}_${OMP}.out"
done

echo "All strong scaling runs complete."
