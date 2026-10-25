#!/usr/bin/env bash
set -e
cd "$HOME/async_llm"
source .venv/bin/activate
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
ln -sfn lib "$CUDA_HOME/lib64" 2>/dev/null || true
mkdir -p "$CUDA_HOME/lib64/stubs"
ln -sfn /usr/lib/wsl/lib/libcuda.so.1 "$CUDA_HOME/lib64/stubs/libcuda.so" 2>/dev/null || true
python /mnt/s/Agent/QSExplore/RecycleLM/scripts/async_correct_demo.py
