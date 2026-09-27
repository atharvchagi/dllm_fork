#!/usr/bin/env bash
# Run the block size 16 baseline evaluation on physical GPUs 4, 5, 6, and 8:
# bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/eval_block16_4gpu.sh

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DLLM_GPUS=GPU-90c9fbaa-d3fb-17aa-b33d-915209e44d3f,GPU-9a62c6db-c881-040b-4a9c-3f0c81475a67,GPU-c31a66b9-cce8-bf56-a3cf-f026d1fdbb6a,GPU-95109861-0d50-d1b0-e10e-b1b16ce19947
export DLLM_MAIN_PROCESS_PORT=29582

exec bash "${script_dir}/eval.sh" 16
