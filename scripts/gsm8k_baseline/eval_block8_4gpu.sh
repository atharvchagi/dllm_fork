#!/usr/bin/env bash
# Run the block size 8 baseline evaluation on physical GPUs 1, 2, 3, and 7:
# bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/eval_block8_4gpu.sh

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DLLM_GPUS=GPU-43f0c2b4-d013-b88a-6595-12ce7fe4ab98,GPU-cb79cdce-7f85-fe4f-6d63-3a0abc3e3929,GPU-af4a1b59-58a5-af18-bdc9-0bf62e636634,GPU-ed6fa8a2-6c94-71dd-60f7-496a84b424bc
export DLLM_MAIN_PROCESS_PORT=29581

exec bash "${script_dir}/eval.sh" 8
