"""Download public WebShop data and build its local Lucene search indexes."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--build-index",action="store_true")
    parser.add_argument("--dataset",default="YWZBrandon/webshop-data")
    args=parser.parse_args()
    if not (args.download or args.build_index):parser.error("Select --download and/or --build-index")
    root=Path(__file__).resolve().parents[1]/"agent_system/environments/env_package/webshop/webshop"
    data=root/"data"
    if args.download:
        from huggingface_hub import hf_hub_download
        data.mkdir(exist_ok=True)
        for name in ["items_shuffle_1000.json","items_ins_v2_1000.json","items_human_ins.json"]:
            hf_hub_download(repo_id=args.dataset,repo_type="dataset",filename=name,local_dir=data)
    if args.build_index:
        search=root/"search_engine"
        for name in ["resources","resources_100","resources_1k","resources_100k"]:
            (search/name).mkdir(exist_ok=True)
        subprocess.run([sys.executable,"convert_product_file_format.py"],cwd=search,check=True)
        for suffix in ["","_100","_1k","_100k"]:
            subprocess.run([sys.executable,"-m","pyserini.index.lucene","--collection","JsonCollection",
                "--input",str(search/f"resources{suffix}"),"--index",str(search/f"indexes{suffix}"),
                "--generator","DefaultLuceneDocumentGenerator","--threads","1",
                "--storePositions","--storeDocvectors","--storeRaw"],check=True)


if __name__=="__main__":main()
