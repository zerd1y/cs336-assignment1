#!/usr/bin/env bash
set -euo pipefail

# AutoDL-oriented TinyStories pipeline:
# 1. preprocess train/valid text into tokenizer + bin files
# 2. train the language model
# 3. generate a 256-token sample
# 4. shut down the machine after successful completion

PROJECT_DIR="${PROJECT_DIR:-/assignment1}"
TRAIN_TXT="${TRAIN_TXT:-/root/autodl-tmp/TinyStories/TinyStories-train.txt}"
VALID_TXT="${VALID_TXT:-/root/autodl-tmp/TinyStories/TinyStories-valid.txt}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/tinystories_data}"
RUN_DIR="${RUN_DIR:-/root/autodl-tmp/tinystories_run}"
PROMPT="${PROMPT:-Once upon a time, a little girl found a secret in the forest.}"

VOCAB_SIZE="${VOCAB_SIZE:-10000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TOTAL_TOKENS="${TOTAL_TOKENS:-327680000}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-3e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.95}"
DEVICE="${DEVICE:-cuda}"
AUTO_SHUTDOWN_ON_SUCCESS="${AUTO_SHUTDOWN_ON_SUCCESS:-1}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

shutdown_server() {
  if [[ "${AUTO_SHUTDOWN_ON_SUCCESS}" != "1" ]]; then
    log "AUTO_SHUTDOWN_ON_SUCCESS=${AUTO_SHUTDOWN_ON_SUCCESS}, skipping shutdown."
    return
  fi

  if command -v shutdown >/dev/null 2>&1; then
    log "Pipeline finished successfully. Shutting down the server now."
    shutdown -h now
  elif command -v poweroff >/dev/null 2>&1; then
    log "Pipeline finished successfully. Powering off the server now."
    poweroff
  else
    log "No shutdown command found. Please stop the server manually."
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
}

main() {
  require_file "${TRAIN_TXT}"
  require_file "${VALID_TXT}"

  cd "${PROJECT_DIR}"

  log "Project directory: ${PROJECT_DIR}"
  log "Train txt: ${TRAIN_TXT}"
  log "Valid txt: ${VALID_TXT}"
  log "Data dir: ${DATA_DIR}"
  log "Run dir: ${RUN_DIR}"

  mkdir -p "${DATA_DIR}" "${RUN_DIR}"

  log "Step 1/3: preprocessing TinyStories text files"
  python -m llm_basics.tinystories preprocess \
    --corpus-path "${TRAIN_TXT}" \
    --dev-corpus-path "${VALID_TXT}" \
    --output-dir "${DATA_DIR}" \
    --vocab-size "${VOCAB_SIZE}" \
    --num-workers "${NUM_WORKERS}"

  log "Step 2/3: training the TinyStories language model"
  python -m llm_basics.tinystories train \
    --data-dir "${DATA_DIR}" \
    --output-dir "${RUN_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --total-tokens "${TOTAL_TOKENS}" \
    --learning-rate "${LEARNING_RATE}" \
    --min-learning-rate "${MIN_LEARNING_RATE}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --grad-clip-norm "${GRAD_CLIP_NORM}" \
    --log-interval "${LOG_INTERVAL}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-batches "${EVAL_BATCHES}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
    --device "${DEVICE}"

  log "Step 3/3: generating a sample text"
  python -m llm_basics.tinystories generate \
    --checkpoint-path "${RUN_DIR}/checkpoint.pt" \
    --tokenizer-path "${DATA_DIR}/tokenizer.json" \
    --prompt "${PROMPT}" \
    --output-path "${RUN_DIR}/sample.txt" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --device "${DEVICE}"

  log "Pipeline completed successfully."
  shutdown_server
}

main "$@"
