#!/usr/bin/env bash
set -euo pipefail
task=${1:-alfworld}
scale=${2:-3b}
if (( $# >= 2 )); then shift 2; else shift "$#"; fi
case "$task" in alfworld|webshop|search) ;; *) echo 'Task must be alfworld, webshop, or search' >&2; exit 2;; esac
case "$scale" in 3b|7b) ;; *) echo 'Scale must be 3b or 7b' >&2; exit 2;; esac
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
exec python -m uniopsd.evaluate --config-name "${task}_corr_${scale}" "$@"
