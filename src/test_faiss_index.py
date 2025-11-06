#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test a FAISS index built from BEIR/SciDocs.

Examples:
  # free-form query string
  uv run python src/test_faiss_index.py \
    --db data/scidocs.duckdb --index_dir indexes/scidocs_faiss \
    --query "what is bert pretraining?" --k 8 --snippet_len 240

  # use a query by query_id (enables qrels-based metrics)
  uv run python src/test_faiss_index.py \
    --db data/scidocs.duckdb --index_dir indexes/scidocs_faiss \
    --query_id 0074e8b3 --k 10 --use_reranker

  # pick the Nth query (offset into scidocs_queries)
  uv run python src/test_faiss_index.py \
    --db data/scidocs.duckdb --index_dir indexes/scidocs_faiss \
    --query_offset 42 --k 10
"""
import argparse
import json
from pathlib import Path
import math
from typing import List, Dict, Tuple

import duckdb
import faiss
import numpy as np
import orjson
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Optional reranker (install torch/transformers if you use --use_reranker)
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _HAS_RERANKER = True
except Exception:
    _HAS_RERANKER = False


def load_meta(meta_path: str) -> Dict[int, Dict]:
    meta = {}
    with open(meta_path, "rb") as f:
        for line in f:
            rec = orjson.loads(line)
            meta[int(rec["faiss_id"])] = rec
    return meta


def ndcg_at_k(relevances: List[float], k: int) -> float:
    """Compute NDCG@k given a list of relevances in ranked order."""
    rel = np.array(relevances[:k], dtype=float)
    # DCG
    gains = (2**rel - 1.0) / np.log2(np.arange(2, rel.size + 2))
    dcg = float(gains.sum())
    # IDCG
    ideal = np.sort(rel)[::-1]
    igains = (2**ideal - 1.0) / np.log2(np.arange(2, ideal.size + 2))
    idcg = float(igains.sum())
    return dcg / idcg if idcg > 0 else 0.0


def quick_stats(scores: List[float]) -> Dict:
    arr = np.array(scores) if len(scores) else np.array([0.0])
    p = arr / (arr.sum() + 1e-9)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "score_mean": float(arr.mean()),
        "score_std": float(arr.std()),
        "score_entropy": ent,
    }


class CrossEncoderReranker:
    def __init__(self, name="BAAI/bge-reranker-base"):
        if not _HAS_RERANKER:
            raise RuntimeError("Reranker unavailable: install torch and transformers.")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()

    @torch.no_grad()
    def rerank(self, query: str, passages: List[Dict], k: int) -> List[Dict]:
        pairs = [(query, p.get("text", "")) for p in passages]
        batch = self.tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        logits = self.model(**batch).logits.squeeze(-1)
        order = torch.argsort(logits, descending=True).tolist()
        order = order[:k]
        return [passages[i] for i in order]


def fetch_query_from_db(con, query_id: str = None, offset: int = None) -> Tuple[str, str]:
    """
    Return (query_text, query_id_or_none)
    """
    if query_id:
        row = con.execute(
            "SELECT query_id, text FROM scidocs_queries WHERE query_id = ?",
            [query_id],
        ).fetchone()
        if not row:
            raise ValueError(f"query_id {query_id} not found in scidocs_queries")
        return row[1], row[0]
    elif offset is not None:
        row = con.execute(
            "SELECT query_id, text FROM scidocs_queries LIMIT 1 OFFSET ?",
            [int(offset)],
        ).fetchone()
        if not row:
            raise ValueError(f"query_offset {offset} beyond end of scidocs_queries")
        return row[1], row[0]
    else:
        raise ValueError("Must pass query_id or query_offset when not using --query")


def fetch_qrels_for_query(con, qid: str) -> Dict[str, int]:
    rows = con.execute(
        "SELECT doc_id, relevance FROM scidocs_qrels WHERE query_id = ?",
        [qid],
    ).fetchall()
    return {r[0]: int(r[1]) for r in rows}

def fetch_doc_texts(con, doc_ids: List[str], snippet_len: int = 0) -> Dict[str, str]:
    if not doc_ids:
        return {}
    # Deduplicate to keep the temp table small
    doc_ids = list(dict.fromkeys(doc_ids))

    try:
        # Preferred path: UNNEST with explicit alias and column name
        df = con.execute(
            """
            WITH ids AS (
              SELECT * FROM UNNEST(?::TEXT[]) AS t(doc_id)
            )
            SELECT d.doc_id, d.title, d.text
            FROM scidocs_docs AS d
            JOIN ids ON d.doc_id = ids.doc_id
            """,
            [doc_ids],
        ).fetch_df()
    except Exception:
        # Fallback path: temp table (works across all DuckDB versions)
        con.execute("CREATE TEMP TABLE tmp_ids(doc_id TEXT)")
        con.executemany("INSERT INTO tmp_ids VALUES (?)", [(x,) for x in doc_ids])
        df = con.execute(
            """
            SELECT d.doc_id, d.title, d.text
            FROM scidocs_docs AS d
            JOIN tmp_ids ON d.doc_id = tmp_ids.doc_id
            """
        ).fetch_df()
        con.execute("DROP TABLE tmp_ids")

    out = {}
    for _, row in df.iterrows():
        title = row["title"] or ""
        body  = row["text"] or ""
        txt = (title + "\n\n" + body).strip()
        if snippet_len and len(txt) > snippet_len:
            txt = txt[:snippet_len] + "…"
        out[row["doc_id"]] = txt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to scidocs.duckdb")
    ap.add_argument("--index_dir", required=True, help="Directory with faiss.index and meta.jsonl")
    ap.add_argument("--emb", default="BAAI/bge-small-en-v1.5", help="Embedding model used to build index")
    ap.add_argument("--k", type=int, default=8, help="Top-k to return")
    ap.add_argument("--query", type=str, help="Free-form query string")
    ap.add_argument("--query_id", type=str, help="Query ID from scidocs_queries (enables qrels metrics)")
    ap.add_argument("--query_offset", type=int, help="Nth query (offset) from scidocs_queries")
    ap.add_argument("--use_reranker", action="store_true", help="Apply cross-encoder reranker")
    ap.add_argument("--reranker_name", default="BAAI/bge-reranker-base")
    ap.add_argument("--snippet_len", type=int, default=220, help="Include doc text snippets (characters); 0 to omit")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    idx_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.jsonl"

    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing {idx_path} or {meta_path}")

    print(f"🔎 Loading index from {idx_path}")
    index = faiss.read_index(str(idx_path))
    print(f"🔎 Loading metadata from {meta_path}")
    meta = load_meta(str(meta_path))
    print(f"🔎 Loading embedder {args.emb}")
    model = SentenceTransformer(args.emb)

    # Query sourcing
    con = duckdb.connect(args.db, read_only=True)
    if args.query:
        query_text, qid = args.query, None
    else:
        query_text, qid = fetch_query_from_db(con, query_id=args.query_id, offset=args.query_offset)

    print(f"\n❓ Query: {query_text}")
    if qid:
        print(f"   id: {qid}")

    # Encode and search
    qv = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    D, I = index.search(qv, max(args.k, 50 if args.use_reranker else args.k))
    candidates = []
    for s, fid in zip(D[0].tolist(), I[0].tolist()):
        rec = meta.get(int(fid))
        if rec:
            candidates.append({
                "faiss_id": int(fid),
                "score": float(s),
                "doc_id": rec.get("doc_id"),
                "title": rec.get("title", ""),
                "year": rec.get("year", 0),
            })

    # Optional rerank
    if args.use_reranker:
        if not _HAS_RERANKER:
            raise RuntimeError("--use_reranker requested but torch/transformers not available")
        rr = CrossEncoderReranker(args.reranker_name)
        # Join text from DB for reranker input
        doc_texts = fetch_doc_texts(con, [c["doc_id"] for c in candidates[:50]], snippet_len=0)
        for c in candidates[:50]:
            c["text"] = doc_texts.get(c["doc_id"], "")
        reranked = rr.rerank(query_text, candidates[:50], k=args.k)
        hits = reranked
    else:
        hits = candidates[:args.k]

    # Pull snippets if requested
    if args.snippet_len > 0:
        doc_texts = fetch_doc_texts(con, [h["doc_id"] for h in hits], snippet_len=args.snippet_len)
        for h in hits:
            h["snippet"] = doc_texts.get(h["doc_id"], "")

    # Diagnostics
    stats = quick_stats([h["score"] for h in hits])
    uniq_titles = len(set(h.get("title", "") for h in hits))
    stats["unique_titles"] = int(uniq_titles)

    # Metrics vs qrels if we know the query id
    retrieval_metrics = {}
    if qid:
        rel_map = fetch_qrels_for_query(con, qid)  # doc_id -> relevance
        rels_in_rank = [float(rel_map.get(h["doc_id"], 0)) for h in hits]
        hits_relevant = sum(1 for r in rels_in_rank if r > 0)
        total_relevant = sum(1 for r in rel_map.values() if r > 0)
        precision = hits_relevant / max(1, args.k)
        recall = hits_relevant / max(1, total_relevant)
        ndcg = ndcg_at_k(rels_in_rank, args.k)
        retrieval_metrics = {
            "relevant@k": hits_relevant,
            "total_relevant": int(total_relevant),
            "precision@k": round(precision, 4),
            "recall@k": round(recall, 4),
            "ndcg@k": round(ndcg, 4),
        }

    # Pretty print
    print("\n=== Top-k results ===")
    for i, h in enumerate(hits, 1):
        line1 = f"{i:>2}. score={h['score']:.4f} | year={h.get('year',0)} | doc_id={h['doc_id']}"
        line2 = f"    title: {h.get('title','')[:220]}"
        print(line1)
        print(line2)
        if args.snippet_len > 0 and h.get("snippet"):
            print(f"    snippet: {h['snippet'].replace('\\n',' ')[:args.snippet_len]}")
    print("\n=== Stats ===")
    print(json.dumps(stats, indent=2))
    if retrieval_metrics:
        print("\n=== Retrieval metrics (vs qrels) ===")
        print(json.dumps(retrieval_metrics, indent=2))

    con.close()


if __name__ == "__main__":
    main()
