#!/bin/bash
# Strong scaling on a single node (no SLURM)
# Intel Xeon Platinum 8480C: 112 physical cores
# Strategy: keep total cores = 64, split between MPI ranks and OMP threads
#
# ranks x omp_threads = 64 (constant)
# 1  rank  x 64 threads
# 2  ranks x 32 threads
# 4  ranks x 16 threads
# 8  ranks x  8 threads
# 16 ranks x  4 threads
# 32 ranks x  2 threads
# 64 ranks x  1 thread

TOTAL_BATCHES=640
BATCH_SIZE=128
EXPORT_DIR=cifar10_embedding_mlp/export
INPUT=cifar10_embedding_mlp/export/test_input_embeddings_all.bin
OUTDIR=benchmark_CPU_Strong

mkdir -p $OUTDIR

for RANKS in 1 2 4 8 16 32 64; do
    OMP=$((64 / RANKS))
    if [ $OMP -lt 1 ]; then OMP=1; fi
    echo "Running: $RANKS ranks x $OMP OMP threads..."

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
