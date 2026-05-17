#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMITMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_env.sh"
PYTHON_BIN="$(resolve_commitment_python_or_die)"
RESULTS_ROOT="${RESULTS_ROOT:-$COMMITMENT_ROOT/results}"

MODEL_NAME=""
GPU_IDS_STR="${GPU_IDS:-}"
RUN_TAG="${RUN_TAG:-}"

MINER_OUTPUT_DIR=""
SENTENCE_DATASET_DIR=""
LOCALIZATION_DIR=""
SENTENCE_DATASET_DIR_EXPLICIT=0
LOCALIZATION_DIR_EXPLICIT=0

N_SAMPLES="${N_SAMPLES:-32}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3000}"
BASE_SEED_START="${BASE_SEED_START:-1234}"
METHOD="${METHOD:-adaptive}"
MODE="${MODE:-prefix}"
TEXT_FIELD="${TEXT_FIELD:-action_reasoning}"
LABEL_FILTER="${LABEL_FILTER:-all}"
LIMIT="${LIMIT:-0}"
LOG_EVERY="${LOG_EVERY:-25}"
COARSE_ITERS="${COARSE_ITERS:-8}"
REFINEMENT_ITERS="${REFINEMENT_ITERS:-8}"
MIN_STEP_SIZE="${MIN_STEP_SIZE:-1}"
MIN_SPACING="${MIN_SPACING:-1}"
TARGET_CORRECT="${TARGET_CORRECT:-0}"
TARGET_INCORRECT="${TARGET_INCORRECT:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
REBALANCE_PENDING_SHARDS="${REBALANCE_PENDING_SHARDS:-1}"
REBUILD_SENTENCE_DATASET="${REBUILD_SENTENCE_DATASET:-0}"
WRITE_JSONL="${WRITE_JSONL:-0}"
JSONL_BASENAME="${JSONL_BASENAME:-localization.jsonl}"

OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  run_sentence_localization_multi_gpu.sh --model_name MODEL --gpu_ids "0 1 2 3" [options]

Required:
  --model_name MODEL          Hugging Face / vLLM model name
  --gpu_ids "0 1 2 3"         Space-separated GPU ids on the current machine

Optional paths:
  --results_root DIR          Default: <commitment>/results
  --run_tag TAG               Reuse a named mining run. If omitted, the latest run is used.
  --miner_output_dir DIR      Explicit commitment miner output dir. Overrides derived lookup.
  --sentence_dataset_dir DIR  Explicit sentence dataset dir. If missing examples/sentences, it will be built.
  --localization_dir DIR      Explicit output dir for per-example localization JSON files.

Optional behavior:
  --text_field FIELD          Default: action_reasoning
  --method adaptive|full      Default: adaptive
  --mode prefix|sentence_only Default: prefix
  --label_filter FILTER       One of: all, correct_only, incorrect_only
  --limit N                   Optional example limit
  --overwrite                 Pass --overwrite to sentence_localization.py
  --write_jsonl               Also write sharded localization JSONL outputs
  --rebuild_sentence_dataset  Force rebuilding examples.jsonl and sentences.jsonl before localization
  --help                      Show this message

Examples:
  bash run_sentence_localization_multi_gpu.sh --model_name MODEL --gpu_ids "0 1 2 3"
  bash run_sentence_localization_multi_gpu.sh --model_name MODEL --gpu_ids "4 5" --miner_output_dir /path/to/mining/run
EOF
}

latest_run_dir() {
  local base_dir="$1"
  if [[ ! -d "$base_dir" ]]; then
    return 1
  fi
  find "$base_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --gpu_ids)
      GPU_IDS_STR="$2"
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
    --miner_output_dir)
      MINER_OUTPUT_DIR="$2"
      shift 2
      ;;
    --sentence_dataset_dir)
      SENTENCE_DATASET_DIR="$2"
      SENTENCE_DATASET_DIR_EXPLICIT=1
      shift 2
      ;;
    --localization_dir)
      LOCALIZATION_DIR="$2"
      LOCALIZATION_DIR_EXPLICIT=1
      shift 2
      ;;
    --text_field)
      TEXT_FIELD="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --label_filter)
      LABEL_FILTER="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --write_jsonl)
      WRITE_JSONL=1
      shift
      ;;
    --rebuild_sentence_dataset)
      REBUILD_SENTENCE_DATASET=1
      shift
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

if [[ -z "$MODEL_NAME" || -z "$GPU_IDS_STR" ]]; then
  usage >&2
  exit 1
fi

GPU_IDS_ARRAY=($GPU_IDS_STR)
NUM_SHARDS=${#GPU_IDS_ARRAY[@]}
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "No GPUs provided." >&2
  exit 1
fi

MODEL_TAG="${MODEL_NAME##*/}"

if [[ -z "$RUN_TAG" ]]; then
  if [[ -n "$SENTENCE_DATASET_DIR" ]]; then
    RUN_TAG="$(basename "$SENTENCE_DATASET_DIR")"
  elif [[ -n "$MINER_OUTPUT_DIR" ]]; then
    RUN_TAG="$(basename "$MINER_OUTPUT_DIR")"
  else
    LATEST_DIR="$(latest_run_dir "$RESULTS_ROOT/mining/$MODEL_TAG" || true)"
    if [[ -n "$LATEST_DIR" ]]; then
      MINER_OUTPUT_DIR="$LATEST_DIR"
      RUN_TAG="$(basename "$LATEST_DIR")"
    fi
  fi
fi

if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="manual"
fi

if [[ -z "$MINER_OUTPUT_DIR" ]]; then
  MINER_OUTPUT_DIR="$RESULTS_ROOT/mining/$MODEL_TAG/$RUN_TAG"
fi

if [[ "$SENTENCE_DATASET_DIR_EXPLICIT" != "1" ]]; then
  SENTENCE_DATASET_DIR="$RESULTS_ROOT/sentence_datasets/$MODEL_TAG/$RUN_TAG"
fi
EXAMPLES_PATH="$SENTENCE_DATASET_DIR/examples.jsonl"
SENTENCES_PATH="$SENTENCE_DATASET_DIR/sentences.jsonl"

if [[ "$LOCALIZATION_DIR_EXPLICIT" == "1" ]]; then
  LOCALIZATION_RUN_ROOT="$(dirname "$LOCALIZATION_DIR")"
else
  LOCALIZATION_RUN_ROOT="$RESULTS_ROOT/localization/$MODEL_TAG/$RUN_TAG"
  LOCALIZATION_DIR="$LOCALIZATION_RUN_ROOT/localization"
fi

mkdir -p "$SENTENCE_DATASET_DIR"
mkdir -p "$LOCALIZATION_DIR"
mkdir -p "$LOCALIZATION_RUN_ROOT"

NEED_SENTENCE_DATASET_BUILD=0
if [[ "$REBUILD_SENTENCE_DATASET" == "1" || ! -f "${EXAMPLES_PATH:-/dev/null}" || ! -f "${SENTENCES_PATH:-/dev/null}" ]]; then
  NEED_SENTENCE_DATASET_BUILD=1
fi

if [[ "$NEED_SENTENCE_DATASET_BUILD" == "1" ]]; then
  if [[ ! -d "$MINER_OUTPUT_DIR" ]]; then
    echo "Need a commitment miner output dir to build the sentence dataset." >&2
    echo "Expected miner output dir: $MINER_OUTPUT_DIR" >&2
    echo "Either run run_commitment_miner_single_gpu.sh first or pass --miner_output_dir /abs/path/to/run." >&2
    exit 1
  fi
  BUILD_CMD=(
    "$PYTHON_BIN" "$COMMITMENT_ROOT/src/build_sentence_dataset.py"
    --input_root "$MINER_OUTPUT_DIR"
    --out_dir "$SENTENCE_DATASET_DIR"
    --text_field "$TEXT_FIELD"
    --label_filter "$LABEL_FILTER"
    --target_correct "$TARGET_CORRECT"
    --target_incorrect "$TARGET_INCORRECT"
  )
  if [[ "$LIMIT" -gt 0 ]]; then
    BUILD_CMD+=(--limit "$LIMIT")
  fi
  echo "Building sentence dataset:"
  printf ' %q' "${BUILD_CMD[@]}"
  printf '\n'
  "${BUILD_CMD[@]}"
fi

if [[ ! -f "$EXAMPLES_PATH" || ! -f "$SENTENCES_PATH" ]]; then
  echo "Sentence dataset build did not produce examples.jsonl / sentences.jsonl." >&2
  exit 1
fi

echo "Model: $MODEL_NAME"
echo "Miner output dir: $MINER_OUTPUT_DIR"
echo "Sentence dataset dir: $SENTENCE_DATASET_DIR"
echo "Localization dir: $LOCALIZATION_DIR"
echo "GPUs: ${GPU_IDS_ARRAY[*]}"
echo "Run tag: $RUN_TAG"
print_commitment_env_notice

pids=()
pid_gpus=()
for idx in "${!GPU_IDS_ARRAY[@]}"; do
  gpu="${GPU_IDS_ARRAY[$idx]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
    export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-/tmp/vllm}"
    mkdir -p "$VLLM_CONFIG_ROOT"

    CMD=(
      "$PYTHON_BIN" "$COMMITMENT_ROOT/src/sentence_localization.py"
      --examples_path "$EXAMPLES_PATH"
      --sentences_path "$SENTENCES_PATH"
      --model_name "$MODEL_NAME"
      --out_dir "$LOCALIZATION_DIR"
      --n_samples "$N_SAMPLES"
      --temperature "$TEMPERATURE"
      --top_p "$TOP_P"
      --repetition_penalty "$REPETITION_PENALTY"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --base_seed "$((BASE_SEED_START + idx * 100000))"
      --method "$METHOD"
      --mode "$MODE"
      --coarse_iters "$COARSE_ITERS"
      --refinement_iters "$REFINEMENT_ITERS"
      --min_step_size "$MIN_STEP_SIZE"
      --min_spacing "$MIN_SPACING"
      --label_filter "$LABEL_FILTER"
      --text_field "$TEXT_FIELD"
      --shard_id "$idx"
      --num_shards "$NUM_SHARDS"
      --log_every "$LOG_EVERY"
      --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
      --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
    )
    if [[ "$LIMIT" -gt 0 ]]; then
      CMD+=(--limit "$LIMIT")
    fi
    if [[ "$OVERWRITE" == "1" ]]; then
      CMD+=(--overwrite)
    fi
    if [[ "$REBALANCE_PENDING_SHARDS" == "1" ]]; then
      CMD+=(--rebalance_pending_shards)
    fi
    if [[ "$WRITE_JSONL" == "1" ]]; then
      CMD+=(--jsonl_path "$LOCALIZATION_RUN_ROOT/$JSONL_BASENAME")
    fi

    echo "--------------------------------"
    echo "Running localization on GPU $gpu"
    printf ' %q' "${CMD[@]}"
    printf '\n'
    echo "--------------------------------"
    "${CMD[@]}" > "$LOCALIZATION_RUN_ROOT/run_gpu_$gpu.log" 2>&1
  ) &
  pids+=("$!")
  pid_gpus+=("$gpu")
done

failed=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    failed=$((failed + 1))
    gpu="${pid_gpus[$idx]}"
    echo "Localization worker failed on GPU $gpu. Tail of log:"
    tail -n 60 "$LOCALIZATION_RUN_ROOT/run_gpu_$gpu.log" || true
  fi
done

if (( failed > 0 )); then
  echo "$failed localization worker(s) failed." >&2
  exit 1
fi

echo "Sentence localization complete."
echo "Localization dir: $LOCALIZATION_DIR"
