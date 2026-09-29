#!/usr/bin/env bash
# Run the released LDDM-U OpenWebText checkpoint on one GPU.

set -euo pipefail

REPO_ROOT=/home/ngarg2/repos/dllm_fork
CONDA_ROOT=/home/ngarg2/miniforge3
MODE=${1:-ppl}
GPU=${LDDM_GPU:-1}
STAMP=$(date +%Y%m%d_%H%M%S)

if [[ -f /home/ngarg2/.zshrc ]]; then
  # shellcheck disable=SC1091
  source /home/ngarg2/.zshrc
else
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ROOT}/envs/dllm"

GPU_UUID=$(nvidia-smi --id="${GPU}" --query-gpu=uuid --format=csv,noheader | sed -n '1p')
if [[ -z "${GPU_UUID}" ]]; then
  echo "Could not resolve physical GPU ${GPU} with nvidia-smi" >&2
  exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_UUID}"
echo "LDDM-U: physical GPU ${GPU} (${GPU_UUID})"
export HF_HOME=${HF_HOME:-${REPO_ROOT}/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${REPO_ROOT}/.cache/huggingface/datasets}
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR=${LDDM_OUTPUT_DIR:-${REPO_ROOT}/results/uniform_loophole/lddm_u_owt/${STAMP}}

extra_args=()
if [[ ${LDDM_GENERATIVE_PPL:-0} == 1 ]]; then
  extra_args+=(--generative-ppl)
fi

cd "${REPO_ROOT}"
python -m dllm.pipelines.lddm_u.eval \
  --mode "${MODE}" \
  --algorithm lddm_u \
  --checkpoint "${REPO_ROOT}/.models/LDDM-U/lddm_u_owt_1M.ckpt" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${REPO_ROOT}/.cache/lddm_u" \
  --dataset "${LDDM_DATASET:-openwebtext-split}" \
  --device cuda:0 \
  --batch-size 1 \
  --length 1024 \
  --steps "${LDDM_STEPS:-1024}" \
  --warmup-batches "${LDDM_WARMUP_BATCHES:-1}" \
  --sample-batches "${LDDM_SAMPLE_BATCHES:-1}" \
  --ppl-batches "${LDDM_PPL_BATCHES:-0}" \
  --ppl-warmup-batches "${LDDM_PPL_WARMUP_BATCHES:-1}" \
  --ppl-mc-samples "${LDDM_PPL_MC_SAMPLES:-1}" \
  --num-workers "${LDDM_NUM_WORKERS:-4}" \
  --wandb-project uniform-loophole \
  --wandb-group owt-uniform-checkpoints \
  --wandb-run-name "lddm-u-owt-${STAMP}" \
  "${extra_args[@]}"
