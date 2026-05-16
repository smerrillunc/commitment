#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMITMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-$COMMITMENT_ROOT/results}"

MODEL_NAME=""
MINER_GPU="${MINER_GPU:-}"
LOCALIZATION_GPUS="${LOCALIZATION_GPUS:-${GPU_IDS:-}}"
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"

usage() {
  cat <<'EOF'
Usage:
  run_commitment_pipeline.sh --model_name MODEL --miner_gpu 0 --localization_gpus "0 1 2 3" [options]

Required:
  --model_name MODEL             Hugging Face / vLLM model name
  --miner_gpu GPU                Single GPU to use for mining
  --localization_gpus "0 1 2"    Space-separated GPUs for localization

Optional:
  --results_root DIR             Default: <commitment>/results
  --run_tag TAG                  Default: timestamp
  --help                         Show this message

The script simply runs:
  1. run_commitment_miner_single_gpu.sh
  2. run_sentence_localization_multi_gpu.sh

Tune miner/localization behavior with environment variables like:
  NUM_QUESTIONS, MAX_SAMPLES_PER_QUESTION, N_SAMPLES, METHOD, MODE, LIMIT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --miner_gpu)
      MINER_GPU="$2"
      shift 2
      ;;
    --localization_gpus)
      LOCALIZATION_GPUS="$2"
      shift 2
      ;;
    --results_root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --run_tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_NAME" || -z "$MINER_GPU" || -z "$LOCALIZATION_GPUS" ]]; then
  usage >&2
  exit 1
fi

echo "Step 1/2: mining commitment traces"
bash "$SCRIPT_DIR/run_commitment_miner_single_gpu.sh" \
  --model_name "$MODEL_NAME" \
  --gpu "$MINER_GPU" \
  --results_root "$RESULTS_ROOT" \
  --run_tag "$RUN_TAG"

echo "Step 2/2: localizing answer accuracy"
bash "$SCRIPT_DIR/run_sentence_localization_multi_gpu.sh" \
  --model_name "$MODEL_NAME" \
  --gpu_ids "$LOCALIZATION_GPUS" \
  --results_root "$RESULTS_ROOT" \
  --run_tag "$RUN_TAG"
