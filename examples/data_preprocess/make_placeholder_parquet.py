"""
Offline replacement for `examples/data_preprocess/prepare.py --mode text`.

Why this exists
---------------
`prepare.py` downloads `hiyouga/geometry3k` from HuggingFace, then throws away
every field it downloaded:

    data = {"data_source": args.mode,
            "prompt": [{"role": "user", "content": ""}],   # instruction_following["text"] == ""
            "ability": "agent",
            "extra_info": {"split": split, "index": idx}}

The dataset only supplies the *row count*; the real ALFWorld/WebShop task
instances are produced by the env at rollout time (this is called out in the
upstream comment in prepare.py). huggingface.co is blocked on this box and
hf-mirror.com is flaky, so we emit the identical rows directly.

Verified equivalent to prepare.py's `text` branch, column-for-column.

Usage:
    python make_placeholder_parquet.py --local_dir <dir> \
        --train_data_size 16 --val_data_size 128
"""

import argparse
import os

import pandas as pd


def build_split(split: str, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "data_source": "text",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "extra_info": {"split": split, "index": idx},
            }
            for idx in range(n)
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=os.path.expanduser("~/data/verl-agent/"))
    parser.add_argument("--mode", default="text", choices=["text"])
    parser.add_argument("--train_data_size", default=16, type=int)
    parser.add_argument("--val_data_size", default=128, type=int)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.expanduser(args.local_dir), args.mode)
    os.makedirs(out_dir, exist_ok=True)

    for split, n, fname in (
        ("train", args.train_data_size, "train.parquet"),
        ("test", args.val_data_size, "test.parquet"),
    ):
        df = build_split(split, n)
        path = os.path.join(out_dir, fname)
        df.to_parquet(path, index=False)
        print(f"wrote {len(df):4d} rows -> {path}")
