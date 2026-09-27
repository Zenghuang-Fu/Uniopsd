#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 vllm==0.11.0
python -m pip install --no-build-isolation flash-attn==2.8.3
python -m pip install -e '.[alfworld,webshop,search,test]'
