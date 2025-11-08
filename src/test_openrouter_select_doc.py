#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select the best document (by index) for a query using an LLM via OpenRouter.

Usage examples:

  # by free-form query
  uv run python src/test_openrouter_select_doc.py \
    --db data/scidocs.duckdb --index_dir indexes/scidocs_faiss \
    --query "A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect" \
    --k 8 --snippet_len 220

  # by BEIR query_offset
  uv run python src/test_openrouter_select_doc.py \
    --db data/scidocs.duckdb --index_dir indexes/scidocs_faiss \
    --query_offset 0 --k 10 --snippet_len 200

  # with reranker + custom model + prompt
  uv run python src/test_openrouter_select_doc.py \
    --db data/scidocs.duckdb --index_dir indexes/scidocs_faiss \
    --query_offset 12 --k 8 --use_reranker \
    --model openai/gpt-5-mini \
    --prompt-path prompts/select_doc/prompt.md
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from jinja2 import Template

import duckdb
import faiss
import numpy as np
import orjson
from sentence_transformers import SentenceTransformer

# Optional reranker (only if you pass --use_reranker)
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _HAS_RERANKER = True
except Exception:
    _HAS_RERANKER = False

# OpenAI SDK pointed at OpenRouter
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("openai python sdk is required: pip install openai>=1.40") from e


def load_meta(meta_path: str) -> Dict[int, Dict]:
    meta = {}
    with open(meta_path, "rb") as f:
        for line in f:
            rec = orjson.loads(line)
            meta[int(rec["faiss_id"])] = rec
    return meta


def fetch_query_from_db(con, query_id: str = None, offset: int = None) -> Tuple[str, str]:
    if query_id:
        row = con.execute(
            "SELECT query_id, text FROM scidocs_queries WHERE query_id = ?", [query_id]
        ).fetchone()
        if not row:
            raise ValueError(f"query_id {query_id} not found")
        return row[1], row[0]
    elif offset is not None:
        row = con.execute(
            "SELECT query_id, text FROM scidocs_queries LIMIT 1 OFFSET ?", [int(offset)]
        ).fetchone()
        if not row:
            raise ValueError(f"query_offset {offset} beyond end")
        return row[1], row[0]
    else:
        raise ValueError("Must pass --query OR (--query_id | --query_offset)")


def fetch_doc_texts(con, doc_ids: List[str], snippet_len: int = 0) -> Dict[str, str]:
    if not doc_ids:
        return {}
    # Deduplicate to keep temp small
    doc_ids = list(dict.fromkeys(doc_ids))
    try:
        df = con.execute(
            """
            WITH ids AS (SELECT * FROM UNNEST(?::TEXT[]) AS t(doc_id))
            SELECT d.doc_id, d.title, d.text
            FROM scidocs_docs AS d
            JOIN ids ON d.doc_id = ids.doc_id
            """,
            [doc_ids],
        ).fetch_df()
    except Exception:
        con.execute("CREATE TEMP TABLE tmp_ids(doc_id TEXT)")
        con.executemany("INSERT INTO tmp_ids VALUES (?)", [(x,) for x in doc_ids])
        df = con.execute(
            "SELECT d.doc_id, d.title, d.text FROM scidocs_docs d JOIN tmp_ids ON d.doc_id = tmp_ids.doc_id"
        ).fetch_df()
        con.execute("DROP TABLE tmp_ids")

    out = {}
    for _, row in df.iterrows():
        title = row["title"] or ""
        body = row["text"] or ""
        txt = (title + "\n\n" + body).strip()
        if snippet_len and len(txt) > snippet_len:
            txt = txt[:snippet_len] + "…"
        out[row["doc_id"]] = txt
    return out


class CrossEncoderReranker:
    def __init__(self, name="BAAI/bge-reranker-base"):
        if not _HAS_RERANKER:
            raise RuntimeError("--use_reranker requested but torch/transformers not available")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()

    @torch.no_grad()
    def rerank(self, query: str, passages: List[Dict], k: int) -> List[Dict]:
        pairs = [(query, p.get("text", "")) for p in passages]
        batch = self.tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        logits = self.model(**batch).logits.squeeze(-1)
        order = torch.argsort(logits, descending=True).tolist()
        return [passages[i] for i in order[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--index_dir", required=True, help="dir with faiss.index and meta.jsonl")
    ap.add_argument("--emb", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--query", type=str)
    ap.add_argument("--query_id", type=str)
    ap.add_argument("--query_offset", type=int)
    ap.add_argument("--use_reranker", action="store_true")
    ap.add_argument("--reranker_name", default="BAAI/bge-reranker-base")
    ap.add_argument("--snippet-len", type=int, default=220)
    ap.add_argument("--model", default="openai/gpt-5-mini",
                    help="OpenRouter model id (e.g., openai/gpt-4o-mini, anthropic/claude-3.5-sonnet)")
    ap.add_argument("--prompt-path", default="prompts/select_doc/prompt.md")
    args = ap.parse_args()

    # ---------- setup retrieval ----------
    index_dir = Path(args.index_dir)
    idx_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.jsonl"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing {idx_path} or {meta_path}")

    con = duckdb.connect(args.db, read_only=True)
    meta = load_meta(str(meta_path))
    index = faiss.read_index(str(idx_path))
    embedder = SentenceTransformer(args.emb)

    # query text
    if args.query:
        query_text, qid = args.query, None
    else:
        query_text, qid = fetch_query_from_db(con, query_id=args.query_id, offset=args.query_offset)

    # search
    qv = embedder.encode([query_text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
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

    # add text for reranker or for prompt snippets
    doc_texts_all = fetch_doc_texts(con, [c["doc_id"] for c in candidates[:50]], snippet_len=args.snippet_len)
    for c in candidates[:50]:
        c["snippet"] = doc_texts_all.get(c["doc_id"], "")

    if args.use_reranker:
        rr = CrossEncoderReranker(args.reranker_name)
        # need full text for reranker (not truncated)
        doc_texts_full = fetch_doc_texts(con, [c["doc_id"] for c in candidates[:50]], snippet_len=0)
        for c in candidates[:50]:
            c["text"] = doc_texts_full.get(c["doc_id"], "")
        reranked = rr.rerank(query_text, candidates[:50], k=args.k)
        hits = reranked
    else:
        hits = candidates[:args.k]

    # ---------- call model via OpenRouter ----------
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Print JSON error so calling tools don't get extra stdout
        print(json.dumps({"index": None, "error": "OPENROUTER_API_KEY not set"}))
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Load prompt template
    prompt_text = Path(args.prompt_path).read_text()
    template = Template(prompt_text)

    # Prepare compact docs JSON for the model
    docs_payload = [
        {
            "index": i,
            "title": h.get("title", "")[:200],
            "doc_id": h.get("doc_id"),
            "snippet": h.get("snippet", "")[:800],
        }
        for i, h in enumerate(hits)
    ]

    # Fill template
    user_content = template.render(
        query=query_text,
        documents_json=json.dumps(docs_payload, ensure_ascii=False)
    )

    try:
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system",
                 "content": "You must respond with valid JSON only. Output exactly one key: index (integer or null)."},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        text = completion.choices[0].message.content
        print(completion)
        print("\n----------------\n")
        print(text)
        # Parse JSON
        obj = json.loads(text)
        # Validate result
        idx = obj.get("index", None)
        if isinstance(idx, int):
            if not (0 <= idx < len(docs_payload)):
                idx = None
        else:
            idx = None if idx is None else None  # force None if not int
        print(json.dumps({"index": idx}))
    except Exception as e:
        print(json.dumps({"index": None, "error": str(e)}))

    con.close()


if __name__ == "__main__":
    main()
