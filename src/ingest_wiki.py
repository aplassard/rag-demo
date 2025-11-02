# src/ingest_wiki.py
from datasets import load_dataset
from pathlib import Path
import argparse, json, orjson

def main(ds_name: str, split: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds = load_dataset(ds_name, "en", split=split, trust_remote_code=True)
    # Expect fields like: title, text, url (dataset-dependent; map as needed)
    with open(Path(out_dir)/"wiki.jsonl", "wb") as f:
        for ex in ds:
            rec = {
                "id": ex.get("id") or ex.get("pageid") or ex["title"],
                "title": ex["title"],
                "text": ex["text"],
                "url": ex.get("url", "")
            }
            f.write(orjson.dumps(rec) + b"\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/finewiki")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/wiki")
    args = ap.parse_args()
    main(args.dataset, args.split, args.out)

