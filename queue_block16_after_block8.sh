#!/usr/bin/env bash
# Run in a separate tmux session:
# /home/ngarg2/repos/dllm_fork/queue_block16_after_block8.sh

set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
train_script="${repo}/scripts/gsm8k_baseline/train.sh"
checkpoint="${repo}/.models/gsm8k_baseline/bd3lm_block8"
log_file="${repo}/.models/gsm8k_baseline/block16.log"
block16_checkpoint="${repo}/.models/gsm8k_baseline/bd3lm_block16"

mkdir -p "${repo}/.models/gsm8k_baseline"
exec 9>"${repo}/.models/gsm8k_baseline/block16.queue.lock"
if ! flock -n 9; then
  echo "A block 16 queue is already running." >&2
  exit 1
fi

echo "Waiting for block 8 training to finish..."
while pgrep -f -x "bash ${train_script} 8" >/dev/null; do
  sleep 30
done

if [[ ! -s "${checkpoint}/model.safetensors" || ! -s "${checkpoint}/tokenizer_config.json" ]]; then
  echo "Block 8 has no final checkpoint; block 16 was not started." >&2
  exit 1
fi
if [[ -s "${block16_checkpoint}/model.safetensors" ]]; then
  echo "Block 16 already has a final checkpoint; nothing to start." >&2
  exit 1
fi

echo "Block 8 finished. Starting block 16; logging to ${log_file}"
bash "${train_script}" 16 2>&1 | tee "${log_file}"
