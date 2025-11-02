# src/build_index_sharded.py
from __future__ import annotations
import os, json, orjson, re
from pathlib import Path
from typing import Iterable, Dict
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def chunk(text, toks=240, stride=160):
    words = re.findall(r"\S+", text or "")
    i=0
    while i < len(words):
        yield " ".join(words[i:i+toks])
        if i + toks >= len(words): break
        i += stride

def read_wiki(path: str) -> Iterable[Dict]:
    with open(path, "rb") as f:
        for line in f:
            yield orjson.loads(line)

def load_manifest(dirpath: Path) -> dict:
    mpath = dirpath / "manifest.json"
    if mpath.exists():
        return json.loads(mpath.read_text())
    return {"embedding_model": None, "dim": None, "next_global_id": 1, "shards": []}

def save_manifest(dirpath: Path, manifest: dict):
    (dirpath / "manifest.json").write_text(json.dumps(manifest, indent=2))

def new_index(dim: int):
    idx = faiss.IndexHNSWFlat(dim, 64)
    idx.hnsw.efConstruction = 120
    return faiss.IndexIDMap(idx)

def build_sharded(
    in_jsonl="data/wiki/wiki.jsonl",
    out_dir="indexes/wiki_faiss",
    emb_model="BAAI/bge-small-en-v1.5",
    max_chunks_per_shard=500_000,
    batch_encode=256,
    resume=True,
    device="mps",
    quantize=None  # options: None | "SQ8" | "PQ64" (see notes below)
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(out)

    model = SentenceTransformer(emb_model)
    dim = model.get_sentence_embedding_dimension()
    if manifest["embedding_model"] is None:
        manifest["embedding_model"] = emb_model
        manifest["dim"] = dim

    # Determine shard numbering
    next_shard_idx = len(manifest["shards"]) + 1

    # Buffers for the current shard
    texts, ids, metas = [], [], []
    shard_count = 0
    next_id = int(manifest.get("next_global_id", 1))

    def flush_shard(shard_name: str):
        nonlocal texts, ids, metas, shard_count
        if not texts: return
        # build index for this shard
        idx = new_index(dim)

        # encode in sub-batches to limit memory
        start = 0
        while start < len(texts):
            sub = texts[start:start+batch_encode]
            embs = model.encode(sub, normalize_embeddings=True, convert_to_numpy=True, batch_size=batch_encode, device=device)
            idx.add_with_ids(embs.astype("float32"), np.array(ids[start:start+batch_encode], dtype="int64"))
            start += batch_encode

        # optional quantization per-shard
        if quantize == "SQ8":
            idx = faiss.index_cpu_to_cpu(idx)  # ensure CPU
            q = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit)
            qi = faiss.IndexPreTransform(q, idx)  # wrapper
            idx = qi
        elif quantize == "PQ64":
            # IVF+PQ factory (replace to taste). Larger IVF list count for big shards.
            idx = faiss.index_cpu_to_cpu(idx)
            idx = faiss.index_factory(dim, "IVF16384,PQ64")
            # You'd need to train IVF/PQ with a sample of vectors: omitted here for brevity.

            # NOTE: If you want IVF/PQ, prefer building the IVF/PQ index directly
            # rather than converting from HNSW. See FAISS docs.

        shard_dir = out / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(idx, str(shard_dir / "faiss.index"))

        with open(shard_dir / "meta.jsonl", "wb") as mf:
            for m in metas: mf.write(orjson.dumps(m) + b"\n")

        # update manifest
        manifest["shards"].append({"name": shard_name, "count": len(metas)})
        manifest["next_global_id"] = next_id
        save_manifest(out, manifest)

        # reset buffers
        shard_count += 1
        texts, ids, metas = [], [], []

    shard_name = f"shard_{next_shard_idx:04d}"

    # If resuming and last shard exists but incomplete, we’ll overwrite that shard safely
    # (Manifest reflects only completed shards, so we’re safe to start a new one.)

    for ex in tqdm(read_wiki(in_jsonl), desc="sharding+chunking"):
        doc_id, title, url = ex.get("id"), ex.get("title",""), ex.get("url","")
        for w in chunk(ex.get("text","")):
            fid = next_id; next_id += 1
            texts.append(w); ids.append(fid)
            metas.append({"faiss_id": fid, "doc_id": doc_id, "title": title, "url": url, "text": w})

            # shard rollover
            if len(texts) >= max_chunks_per_shard:
                flush_shard(shard_name)
                next_shard_idx += 1
                shard_name = f"shard_{next_shard_idx:04d}"

    # final flush
    flush_shard(shard_name)
    print(f"✅ Finished. Shards: {len(manifest['shards'])}, next_global_id={manifest['next_global_id']}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", default="data/wiki/wiki.jsonl")
    ap.add_argument("--out-dir", default="indexes/wiki_faiss")
    ap.add_argument("--emb-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max-chunks-per-shard", type=int, default=500_000)
    ap.add_argument("--batch-encode", type=int, default=256)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quantize", choices=["SQ8","PQ64"], default=None)
    ap.add_argument("--device", choices=["cpu","mps", "gpu"], default="cpu")
    args = ap.parse_args()
    build_sharded(**vars(args))
