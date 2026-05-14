#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="../results"
mkdir -p "$RESULTS_DIR"

THREADS="1 2 4 8 16 32"

EXPORT_DIR="../cifar10_embedding_mlp/export"
INPUT_FILE="../cifar10_embedding_mlp/export/test_input_embeddings_all.bin"
BATCH_SIZE=156

PERF_EVENTS="cache-misses,cache-references,LLC-load-misses,LLC-loads"

MODE="${1:-all}"
VARIANT="${2:-}"
BIN="${3:-./mlp_infer_avx2_max_openmp}"
RUN_TYPE="${4:-perf}"   # perf or basic

run_test() {
  local name="$1"

  if [ -n "$VARIANT" ]; then
    name="${name}_${VARIANT}"
  fi

  name="${name}_${RUN_TYPE}"

  local outfile="$RESULTS_DIR/${name}.txt"

  if [ -e "$outfile" ]; then
    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    outfile="$RESULTS_DIR/${name}_${timestamp}.txt"
  fi

  shift

  echo "Writing results to: $outfile"

  for t in $THREADS; do
    echo "Running $name with OMP_NUM_THREADS=$t"
    echo "=== OMP_NUM_THREADS=$t ===" >> "$outfile"

    if [ "$RUN_TYPE" = "perf" ]; then
      env "$@" OMP_NUM_THREADS="$t" \
        perf stat -e "$PERF_EVENTS" \
        "$BIN" \
        --export-dir "$EXPORT_DIR" \
        --input "$INPUT_FILE" \
        --batch-size "$BATCH_SIZE" \
        --quiet \
        >> "$outfile" 2>> "$outfile"

    elif [ "$RUN_TYPE" = "basic" ]; then
      env "$@" OMP_NUM_THREADS="$t" \
        "$BIN" \
        --export-dir "$EXPORT_DIR" \
        --input "$INPUT_FILE" \
        --batch-size "$BATCH_SIZE" \
        --quiet \
        >> "$outfile" 2>> "$outfile"

    else
      echo "Unknown run type: $RUN_TYPE"
      echo "Use: perf or basic"
      exit 1
    fi

    echo "" >> "$outfile"
  done
}

run_threads_only() {
  run_test "avx2_max_openmp_threads-only"
}

run_bind_true() {
  run_test "avx2_max_openmp_bind-true_places-cores" \
    OMP_PROC_BIND=true \
    OMP_PLACES=cores
}

run_bind_close() {
  run_test "avx2_max_openmp_bind-close_places-cores" \
    OMP_PROC_BIND=close \
    OMP_PLACES=cores
}

run_bind_spread() {
  run_test "avx2_max_openmp_bind-spread_places-cores" \
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
    echo "Usage: $0 [all|threads-only|bind-true|bind-close|bind-spread] [variant] [binary] [perf|basic]"
    exit 1
    ;;
esac