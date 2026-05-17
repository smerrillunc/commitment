#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMITMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_env.sh"
PYTHON_BIN="$(resolve_commitment_python_or_die)"
RESULTS_ROOT="${RESULTS_ROOT:-$COMMITMENT_ROOT/results}"

MODEL_NAME=""
GPU_ID="${CUDA_VISIBLE_DEVICES:-}"
OUTPUT_DIR=""
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"

DATASET_NAME="${DATASET_NAME:-math_qa}"
SPLIT="${SPLIT:-test}"
NUM_QUESTIONS="${NUM_QUESTIONS:-100}"
SAMPLES_PER_ROUND="${SAMPLES_PER_ROUND:-4}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-16}"
MAX_SAMPLES_PER_QUESTION="${MAX_SAMPLES_PER_QUESTION:-40}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-3000}"
SEED="${SEED:-7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
USE_CHAT_TEMPLATE=1
RESUME=1
TRUST_REMOTE_CODE=1

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_commitment_miner_single_gpu.sh --model_name MODEL [options] [-- extra args]

Required:
  --model_name MODEL            Hugging Face / vLLM model name

Optional:
  --gpu GPU                     Optional single GPU id; otherwise uses existing CUDA_VISIBLE_DEVICES
  --results_root DIR            Default: <commitment>/results
  --output_dir DIR              Optional explicit miner output dir
  --run_tag TAG                 Default: timestamp
  --dataset_name NAME           Default: math_qa
  --split SPLIT                 Default: test
  --num_questions N             Default: 100
  --samples_per_round N         Default: 4
  --prompt_batch_size N         Default: 16
  --max_samples_per_question N  Default: 40
  --temperature FLOAT           Default: 0.7
  --top_p FLOAT                 Default: 0.95
  --max_tokens N                Default: 3000
  --seed N                      Default: 7
  --no_resume                   Disable resuming from existing commitment_samples.jsonl
  --no_chat_template            Disable tokenizer chat template rendering
  --no_trust_remote_code        Disable trust_remote_code for dataset/model loading
  --help                        Show this message

Examples:
  bash run_commitment_miner_single_gpu.sh --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --gpu 0
  bash run_commitment_miner_single_gpu.sh --model_name MODEL --output_dir /tmp/commitment_run --num_questions 200

Anything after `--` is forwarded directly to src/commitment_miner.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --results_root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run_tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --dataset_name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --num_questions)
      NUM_QUESTIONS="$2"
      shift 2
      ;;
    --samples_per_round)
      SAMPLES_PER_ROUND="$2"
      shift 2
      ;;
    --prompt_batch_size)
      PROMPT_BATCH_SIZE="$2"
      shift 2
      ;;
    --max_samples_per_question)
      MAX_SAMPLES_PER_QUESTION="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top_p)
      TOP_P="$2"
      shift 2
      ;;
    --max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --no_resume)
      RESUME=0
      shift
      ;;
    --no_chat_template)
      USE_CHAT_TEMPLATE=0
      shift
      ;;
    --no_trust_remote_code)
      TRUST_REMOTE_CODE=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_NAME" ]]; then
  usage >&2
  exit 1
fi

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

MODEL_TAG="${MODEL_NAME##*/}"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$RESULTS_ROOT/mining/$MODEL_TAG/$RUN_TAG"
fi
mkdir -p "$OUTPUT_DIR"

CMD=(
  "$PYTHON_BIN" "$COMMITMENT_ROOT/src/commitment_miner.py"
  --model_name "$MODEL_NAME"
  --output_dir "$OUTPUT_DIR"
  --dataset_name "$DATASET_NAME"
  --split "$SPLIT"
  --num_questions "$NUM_QUESTIONS"
  --samples_per_round "$SAMPLES_PER_ROUND"
  --prompt_batch_size "$PROMPT_BATCH_SIZE"
  --max_samples_per_question "$MAX_SAMPLES_PER_QUESTION"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --max_tokens "$MAX_TOKENS"
  --seed "$SEED"
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
)

if [[ "$USE_CHAT_TEMPLATE" == "1" ]]; then
  CMD+=(--use_chat_template)
else
  CMD+=(--no_use_chat_template)
fi

if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
else
  CMD+=(--no_resume)
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  CMD+=(--trust_remote_code)
else
  CMD+=(--no_trust_remote_code)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Model: $MODEL_NAME"
echo "Output dir: $OUTPUT_DIR"
print_commitment_env_notice
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo "Running:"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
