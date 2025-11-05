# src/build_index_sharded.py
from __future__ import annotations
import os, json, orjson, re, time
from pathlib import Path
from typing import Iterable, Dict
import faiss, numpy as np
import torch
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
    compute_dtype="auto",
    quantize=None  # options: None | "SQ8" | "PQ64" (see notes below)
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(out)

    normalized_device = {
        "gpu": "cuda",
        "cuda": "cuda",
        "cpu": "cpu",
        "mps": "mps",
    }.get(str(device).lower(), str(device))

    if normalized_device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available. Falling back to CPU.")
        normalized_device = "cpu"
    if normalized_device == "mps" and not torch.backends.mps.is_available():
        print("⚠️ MPS requested but not available. Falling back to CPU.")
        normalized_device = "cpu"

    dtype_map = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    compute_dtype_key = str(compute_dtype).lower()
    if compute_dtype_key not in dtype_map:
        raise ValueError(f"Unsupported compute_dtype '{compute_dtype}'. Choose from {list(dtype_map.keys())}.")

    chosen_dtype = dtype_map[compute_dtype_key]
    if chosen_dtype is None:
        if normalized_device in ("cuda", "mps") or normalized_device.startswith("cuda"):
            chosen_dtype = torch.float16
        else:
            chosen_dtype = torch.float32

    if normalized_device == "cpu" and chosen_dtype != torch.float32:
        print("⚠️ Non-fp32 dtype requested on CPU; defaulting to float32.")
        chosen_dtype = torch.float32
    if normalized_device == "mps" and chosen_dtype == torch.bfloat16:
        print("⚠️ bfloat16 not supported on MPS; defaulting to float16.")
        chosen_dtype = torch.float16

    model_kwargs = {"dtype": chosen_dtype}
    model = SentenceTransformer(emb_model, device=normalized_device, model_kwargs=model_kwargs)
    model = model.to(dtype=chosen_dtype)
    model.eval()
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
    total_chunks = 0
    total_docs = 0

    def flush_shard(shard_name: str):
        nonlocal texts, ids, metas, shard_count, total_chunks
        if not texts: return
        # build index for this shard
        idx = new_index(dim)
        shard_chunk_count = len(texts)
        print(
            f"[shard {shard_name}] building index for {shard_chunk_count} chunks "
            f"(processed {total_chunks + shard_chunk_count} total)",
            flush=True,
        )

        effective_batch_size = max(1, min(batch_encode, shard_chunk_count))
        encode_start = time.perf_counter()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=effective_batch_size,
            device=normalized_device,
            show_progress_bar=shard_chunk_count >= effective_batch_size * 8,
        )
        encode_duration = time.perf_counter() - encode_start
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        id_array = np.ascontiguousarray(ids, dtype="int64")
        add_start = time.perf_counter()
        idx.add_with_ids(embeddings, id_array)
        add_duration = time.perf_counter() - add_start

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
        write_start = time.perf_counter()
        faiss.write_index(idx, str(shard_dir / "faiss.index"))
        write_duration = time.perf_counter() - write_start

        with open(shard_dir / "meta.jsonl", "wb") as mf:
            for m in metas: mf.write(orjson.dumps(m) + b"\n")

        # update manifest
        manifest["shards"].append({"name": shard_name, "count": len(metas)})
        manifest["next_global_id"] = next_id
        save_manifest(out, manifest)

        # reset buffers
        shard_count += 1
        total_chunks += shard_chunk_count
        texts, ids, metas = [], [], []
        print(
            f"[shard {shard_name}] encode {encode_duration:.2f}s "
            f"({shard_chunk_count/encode_duration if encode_duration else 0:.0f} chunks/s), "
            f"add {add_duration:.2f}s, write {write_duration:.2f}s",
            flush=True,
        )

    shard_name = f"shard_{next_shard_idx:04d}"

    # If resuming and last shard exists but incomplete, we’ll overwrite that shard safely
    # (Manifest reflects only completed shards, so we’re safe to start a new one.)

    for ex in tqdm(read_wiki(in_jsonl), desc="sharding+chunking"):
        doc_id, title, url = ex.get("id"), ex.get("title",""), ex.get("url","")
        total_docs += 1
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
    print(
        f"✅ Finished. Shards: {len(manifest['shards'])}, "
        f"chunks: {total_chunks}, docs read: {total_docs}, "
        f"next_global_id={manifest['next_global_id']}",
    )

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
    ap.add_argument("--device", choices=["cpu","mps","cuda","gpu"], default="cpu")
    ap.add_argument("--compute-dtype", choices=["auto","float32","float16","bfloat16"], default="auto")
    args = ap.parse_args()
    build_sharded(**vars(args))
