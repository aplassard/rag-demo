#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a FAISS index from SciDocs docs in DuckDB.

Usage:
  python build_faiss_index_from_duckdb.py \
      --db data/scidocs.duckdb \
      --out indexes/scidocs_faiss \
      --emb BAAI/bge-small-en-v1.5 \
      --batch 256 \
      --hnsw_m 32 \
      --hnsw_efc 80
"""
import argparse
import os
from pathlib import Path
import duckdb
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import orjson

def new_hnsw_index(dim: int, M: int = 32, efc: int = 80):
    idx = faiss.IndexHNSWFlat(dim, M)
    idx.hnsw.efConstruction = efc
    return faiss.IndexIDMap(idx)

def main(db_path: str, out_dir: str, emb_model: str, batch: int, hnsw_m: int, hnsw_efc: int):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    idx_path = Path(out_dir) / "faiss.index"
    meta_path = Path(out_dir) / "meta.jsonl"

    con = duckdb.connect(db_path, read_only=True)
    total_docs = con.execute("SELECT COUNT(*) FROM scidocs_docs").fetchone()[0]
    print(f"Found {total_docs} docs in scidocs_docs")

    model = SentenceTransformer(emb_model)
    dim = model.get_sentence_embedding_dimension()

    faiss.omp_set_num_threads(int(os.getenv("FAISS_OMP_THREADS", "8")))
    index = new_hnsw_index(dim, M=hnsw_m, efc=hnsw_efc)

    # Stream in pages
    page_size = 5000
    written = 0
    next_faiss_id = 1
    with open(meta_path, "wb") as mf:
        for offset in range(0, total_docs, page_size):
            df = con.execute(
                "SELECT doc_id, title, text, year FROM scidocs_docs LIMIT ? OFFSET ?",
                [page_size, offset]
            ).fetch_df()

            # prepare texts
            docs = df.to_dict("records")
            texts, ids = [], []
            metas = []
            for i, d in enumerate(docs):
                rid = next_faiss_id; next_faiss_id += 1
                title = d["title"] or ""
                body  = d["text"] or ""
                txt = (title + "\n\n" + body).strip()
                texts.append(txt)
                ids.append(rid)
                metas.append({
                    "faiss_id": rid,
                    "doc_id": d["doc_id"],
                    "title": title,
                    "year": int(d["year"]) if d["year"] is not None else 0
                })

            # embed in mini-batches to control memory
            for start in tqdm(range(0, len(texts), batch), desc=f"embed+add offset={offset}", leave=False):
                sub = texts[start:start+batch]
                sub_ids = ids[start:start+batch]
                embs = model.encode(
                    sub,
                    batch_size=min(batch, 512),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).astype("float32")
                index.add_with_ids(embs, np.array(sub_ids, dtype="int64"))
                written += len(sub)

            # write metas buffered per page
            mf.write(b"".join(orjson.dumps(m) + b"\n" for m in metas))

    faiss.write_index(index, str(idx_path))
    con.close()
    print(f"✅ Wrote index: {idx_path} and meta: {meta_path} (vectors: {written})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/scidocs.duckdb")
    ap.add_argument("--out", default="indexes/scidocs_faiss")
    ap.add_argument("--emb", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hnsw_m", type=int, default=32)
    ap.add_argument("--hnsw_efc", type=int, default=80)
    args = ap.parse_args()
    main(args.db, args.out, args.emb, args.batch, args.hnsw_m, args.hnsw_efc)
