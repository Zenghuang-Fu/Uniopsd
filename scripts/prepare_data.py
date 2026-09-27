"""Prepare public task data; no model service or credentials are configured here."""
import argparse
from pathlib import Path


def agent_data(args):
    import pandas as pd
    output=Path(args.output)/"text"
    output.mkdir(parents=True,exist_ok=True)
    for split,count in (("train",args.train_size),("test",args.val_size)):
        rows=[{"data_source":"text", "prompt":[{"role":"user","content":""}],
               "ability":"agent", "extra_info":{"split":split,"index":i}} for i in range(count)]
        pd.DataFrame(rows).to_parquet(output/f"{split}.parquet",index=False)
    print(f"Prepared task placeholders in {output}")


def search_data(args):
    import importlib.util
    from types import SimpleNamespace
    script=Path(__file__).resolve().parents[1]/"examples/data_preprocess/preprocess_search_r1_dataset.py"
    spec=importlib.util.spec_from_file_location("search_preprocess",script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.args=SimpleNamespace(local_dir=args.output,hdfs_dir=None,hf_repo_id=args.dataset)
    module.system_content=module.DEFAULT_SYSTEM_CONTENT
    module.user_content_prefix=module.DEFAULT_USER_CONTENT_PREFIX
    module.main()
    for split in ("train","test"):
        if not (Path(args.output)/f"{split}.parquet").is_file():
            raise RuntimeError(f"Search preprocessing did not produce {split}.parquet")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="task",required=True)
    agent=sub.add_parser("agent",help="ALFWorld/WebShop task placeholder rows")
    agent.add_argument("--output",default="data/agent")
    agent.add_argument("--train-size",type=int,default=16)
    agent.add_argument("--val-size",type=int,default=128)
    search=sub.add_parser("search",help="Public Natural Questions/HotpotQA data")
    search.add_argument("--output",default="data/search")
    search.add_argument("--dataset",default="PeterJinGo/nq_hotpotqa_train")
    args=parser.parse_args()
    if args.task=="agent":
        if args.train_size<1 or args.val_size<1:parser.error("Dataset sizes must be positive")
        agent_data(args)
    else:search_data(args)


if __name__=="__main__":main()
