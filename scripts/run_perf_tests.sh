#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="../results"
mkdir -p "$RESULTS_DIR"

THREADS="1 2 4 8 16 32"

BIN="./mlp_infer_avx2_max"
EXPORT_DIR="../cifar10_embedding_mlp/export"
INPUT_FILE="../cifar10_embedding_mlp/export/test_input_embeddings_all.bin"
BATCH_SIZE=156

PERF_EVENTS="cache-misses,cache-references,LLC-load-misses,LLC-loads"

MODE="${1:-all}"

run_test() {
  local name="$1"
  local outfile="$RESULTS_DIR/${name}.txt"

  shift

  echo "Writing results to: $outfile"
  : > "$outfile"

  for t in $THREADS; do
    echo "Running $name with OMP_NUM_THREADS=$t"
    echo "=== OMP_NUM_THREADS=$t ===" >> "$outfile"

    env "$@" OMP_NUM_THREADS="$t" \
      perf stat -e "$PERF_EVENTS" \
      "$BIN" \
      --export-dir "$EXPORT_DIR" \
      --input "$INPUT_FILE" \
      --batch-size "$BATCH_SIZE" \
      > /dev/null 2>> "$outfile"

    echo "" >> "$outfile"
  done
}

run_threads_only() {
  run_test "avx2_max_openmp_perf_threads-only"
}

run_bind_true() {
  run_test "avx2_max_openmp_perf_bind-true_places-cores" \
    OMP_PROC_BIND=true \
    OMP_PLACES=cores
}

run_bind_close() {
  run_test "avx2_max_openmp_perf_bind-close_places-cores" \
    OMP_PROC_BIND=close \
    OMP_PLACES=cores
}

run_bind_spread() {
  run_test "avx2_max_openmp_perf_bind-spread_places-cores" \
    OMP_PROC_BIND=spread \
    OMP_PLACES=cores
}

case "$MODE" in
  all)
    run_threads_only
    run_bind_true
    run_bind_close
    run_bind_spread
    ;;
  threads-only)
    run_threads_only
    ;;
  bind-true)
    run_bind_true
    ;;
  bind-close)
    run_bind_close
    ;;
  bind-spread)
    run_bind_spread
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [all|threads-only|bind-true|bind-close|bind-spread]"
    exit 1
    ;;
esac