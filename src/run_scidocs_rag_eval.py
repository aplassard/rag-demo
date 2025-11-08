#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run full BEIR/SciDocs RAG selection against an OpenRouter model and log results to DuckDB.

Creates (or appends to) a DuckDB DB with a results table:

  scidocs_rag_results(
      run_id TEXT,                          -- tag for this run
      ts TIMESTAMP,                         -- when row logged
      model TEXT,                           -- e.g., openai/gpt-5
      reasoning JSON,                       -- the reasoning config sent
      query_id TEXT,
      query_text TEXT,
      k INTEGER,
      retrieved JSON,                       -- [{doc_id,title,score}, ...] (top-k after rerank)
      response_json JSON,                   -- raw parsed json from model (ideally {"index":...})
      response_text TEXT,                   -- raw content returned by model (string)
      selected_index INTEGER,               -- selected index (nullable)
      selected_doc_id TEXT,                 -- doc_id (nullable)
      parse_error TEXT,                     -- parse failure message, if any
      any_relevant_in_topk BOOLEAN,         -- from qrels
      correct BOOLEAN,                      -- selected doc is relevant (or null if no relevant in top-k and model returned null)
      usage_prompt_tokens INTEGER,
      usage_reasoning_tokens INTEGER,       -- if model/provider returns it
      usage_completion_tokens INTEGER,
      latency_ms DOUBLE
  )

Usage (example):
  uv run --env-file .env python src/run_scidocs_rag_eval.py \
    --db data/scidocs.duckdb \
    --results_db data/scidocs_results.duckdb \
    --index_dir indexes/scidocs_faiss \
    --model openai/gpt-5 \
    --k 10 \
    --prompt_path prompts/select_doc/prompt.md.jinja \
    --threads 4 \
    --reasoning_effort medium

OpenRouter reasoning params (unified):
  - --reasoning_effort {low,medium,high}  # OpenAI/Grok style
  - --reasoning_max_tokens INT            # Anthropic/Gemini style
  - --reasoning_enabled                   # turn on with defaults
  - --reasoning_exclude                   # think but don't return traces

Docs:
  - Unified OpenRouter reasoning params: https://openrouter.ai/docs/use-cases/reasoning-tokens
  - Usage accounting / token fields: https://openrouter.ai/docs/use-cases/usage-accounting
"""

import argparse, os, time, json, orjson
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import duckdb
import faiss
import numpy as np
from tqdm import tqdm
from jinja2 import Template
from sentence_transformers import SentenceTransformer

# Optional reranker (enable with --use_reranker)
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _HAS_RERANKER = True
except Exception:
    _HAS_RERANKER = False

# OpenAI SDK pointed at OpenRouter
from openai import OpenAI


# ---------- helpers ----------
def load_meta(meta_path: str) -> Dict[int, Dict]:
    meta = {}
    with open(meta_path, "rb") as f:
        for line in f:
            rec = orjson.loads(line)
            meta[int(rec["faiss_id"])] = rec
    return meta

def fetch_scidocs_queries(con) -> List[Tuple[str, str]]:
    df = con.execute("SELECT query_id, text FROM scidocs_queries").fetch_df()
    return [(r["query_id"], r["text"]) for _, r in df.iterrows()]

def fetch_qrels_map(con) -> Dict[str, Dict[str, int]]:
    # query_id -> {doc_id: relevance}
    rows = con.execute("SELECT query_id, doc_id, relevance FROM scidocs_qrels").fetchall()
    by_q = {}
    for qid, did, rel in rows:
        d = by_q.setdefault(qid, {})
        d[did] = int(rel)
    return by_q

def fetch_docs_texts(con, doc_ids: List[str], snippet_len: int = 220) -> Dict[str, str]:
    if not doc_ids:
        return {}
    doc_ids = list(dict.fromkeys(doc_ids))
    try:
        df = con.execute(
            """
            WITH ids AS (SELECT * FROM UNNEST(?::TEXT[]) AS t(doc_id))
            SELECT d.doc_id, d.title, d.text FROM scidocs_docs d
            JOIN ids ON d.doc_id = ids.doc_id
            """,
            [doc_ids],
        ).fetch_df()
    except Exception:
        # fallback for older duckdb
        con.execute("CREATE TEMP TABLE tmp_ids(doc_id TEXT)")
        con.executemany("INSERT INTO tmp_ids VALUES (?)", [(x,) for x in doc_ids])
        df = con.execute(
            "SELECT d.doc_id, d.title, d.text FROM scidocs_docs d JOIN tmp_ids ON d.doc_id = tmp_ids.doc_id"
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

class CrossEncoderReranker:
    def __init__(self, name="BAAI/bge-reranker-base"):
        if not _HAS_RERANKER:
            raise RuntimeError("Reranker requires torch+transformers")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()

    @torch.no_grad()
    def rerank(self, query: str, passages: List[Dict], k: int) -> List[Dict]:
        pairs = [(query, p.get("full_text", "")) for p in passages]
        batch = self.tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        logits = self.model(**batch).logits.squeeze(-1)
        order = torch.argsort(logits, descending=True).tolist()
        return [passages[i] for i in order[:k]]

def ensure_results_table(con, results_db: str):
    con.execute("""
    CREATE TABLE IF NOT EXISTS scidocs_rag_results(
        run_id TEXT,
        ts TIMESTAMP,
        model TEXT,
        reasoning JSON,
        query_id TEXT,
        query_text TEXT,
        k INTEGER,
        retrieved JSON,
        response_json JSON,
        response_text TEXT,
        selected_index INTEGER,
        selected_doc_id TEXT,
        parse_error TEXT,
        any_relevant_in_topk BOOLEAN,
        correct BOOLEAN,
        usage_prompt_tokens INTEGER,
        usage_reasoning_tokens INTEGER,
        usage_completion_tokens INTEGER,
        latency_ms DOUBLE
    );
    """)

def openrouter_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def build_reasoning_config(args) -> Optional[dict]:
    # OpenRouter's unified 'reasoning' param. You can pass either effort OR max_tokens,
    # plus exclude/enabled. (Docs)  [oai_citation:1‡OpenRouter](https://openrouter.ai/docs/use-cases/reasoning-tokens)
    rc = {}
    if args.reasoning_effort:
        rc["effort"] = args.reasoning_effort  # low|medium|high (OpenAI/Grok)  [oai_citation:2‡OpenRouter](https://openrouter.ai/docs/use-cases/reasoning-tokens)
    if args.reasoning_max_tokens is not None:
        rc["max_tokens"] = int(args.reasoning_max_tokens)  # Anthropic/Gemini thinking budget  [oai_citation:3‡OpenRouter](https://openrouter.ai/docs/use-cases/reasoning-tokens)
    if args.reasoning_exclude:
        rc["exclude"] = True
    if args.reasoning_enabled:
        rc["enabled"] = True
    return rc if rc else None

# ---------- per-query worker ----------
def process_query(qitem, model_name, client, index, index_lock, meta, embedder,
                  con, qrels_map, k, prompt_tmpl: Template, snippet_len,
                  use_reranker, reranker_name, reasoning_cfg) -> dict:
    qid, qtext = qitem
    t0 = time.time()

    # Encode & search (serialize FAISS search if sharing the same object across threads)
    qv = embedder.encode([qtext], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    with index_lock:
        D, I = index.search(qv, max(k, 50 if use_reranker else k))

    # Gather candidates
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

    # Add text for reranking/prompt
    doc_texts_full = fetch_docs_texts(con, [c["doc_id"] for c in candidates[:50]], snippet_len=0)
    doc_texts_snip = fetch_docs_texts(con, [c["doc_id"] for c in candidates[:50]], snippet_len=snippet_len)
    for c in candidates[:50]:
        c["full_text"] = doc_texts_full.get(c["doc_id"], "")
        c["snippet"]   = doc_texts_snip.get(c["doc_id"], "")

    if use_reranker:
        rr = CrossEncoderReranker(reranker_name)
        hits = rr.rerank(qtext, candidates[:50], k=k)
    else:
        hits = candidates[:k]

    # Build docs payload for the prompt (index within current list)
    docs_payload = [
        {
            "index": i,
            "title": h.get("title", "")[:200],
            "doc_id": h.get("doc_id"),
            "snippet": h.get("snippet", "")[:1200],
        }
        for i, h in enumerate(hits)
    ]

    # Render prompt (Jinja2)
    user_content = prompt_tmpl.render(
        query=qtext,
        documents_json=json.dumps(docs_payload, ensure_ascii=False)
    )

    # Compose request
    messages = [
        {"role": "system", "content": "Respond ONLY with JSON: {\"index\": <int or null>}."},
        {"role": "user", "content": user_content},
    ]
    extra_body = {}
    if reasoning_cfg:
        # Pass under extra_body to preserve OpenAI SDK schema.  [oai_citation:4‡OpenRouter](https://openrouter.ai/docs/use-cases/reasoning-tokens)
        extra_body["reasoning"] = reasoning_cfg

    # Call model
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            extra_body=extra_body if extra_body else None,
        )
        content = resp.choices[0].message.content or ""
        parsed = None
        parse_error = None
        try:
            parsed = json.loads(content)
        except Exception as pe:
            parse_error = f"json_parse_error: {pe}"

        # usage accounting (if provider returns it)  [oai_citation:5‡OpenRouter](https://openrouter.ai/docs/use-cases/usage-accounting?utm_source=chatgpt.com)
        usage = getattr(resp, "usage", None) or {}
        prompt_toks     = getattr(usage, "prompt_tokens", None) or usage.get("prompt_tokens")
        completion_toks = getattr(usage, "completion_tokens", None) or usage.get("completion_tokens")
        reasoning_toks  = usage.get("reasoning_tokens") if isinstance(usage, dict) else getattr(usage, "reasoning_tokens", None)

        # selection
        sel_index = None
        sel_doc_id = None
        if isinstance(parsed, dict) and "index" in parsed and parsed["index"] is not None:
            try:
                idx = int(parsed["index"])
                if 0 <= idx < len(docs_payload):
                    sel_index = idx
                    sel_doc_id = docs_payload[idx]["doc_id"]
                else:
                    parse_error = (parse_error or "") + f" invalid_index:{parsed['index']}"
            except Exception as e:
                parse_error = (parse_error or "") + f" cast_index_error:{e}"

        # correctness vs qrels
        relmap = qrels_map.get(qid, {})
        any_rel_in_topk = any(relmap.get(h["doc_id"], 0) > 0 for h in hits)
        is_correct = None
        if sel_doc_id is not None:
            is_correct = bool(relmap.get(sel_doc_id, 0) > 0)
        elif not any_rel_in_topk:
            # If no relevant in top-k and model returned null, that's acceptable; mark False or None?
            # We'll mark False for strict correctness, but you can pivot later.
            is_correct = False

        latency_ms = (time.time() - t0) * 1000.0

        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "reasoning": json.dumps(reasoning_cfg) if reasoning_cfg else None,
            "query_id": qid,
            "query_text": qtext,
            "k": int(k),
            "retrieved": json.dumps([{"doc_id":h["doc_id"],"title":h["title"],"score":h["score"]} for h in hits]),
            "response_json": content,  # the raw JSON string from model (already JSON)
            "response_text": content,  # same (kept for symmetry / future changes)
            "selected_index": sel_index,
            "selected_doc_id": sel_doc_id,
            "parse_error": parse_error,
            "any_relevant_in_topk": any_rel_in_topk,
            "correct": is_correct,
            "usage_prompt_tokens": int(prompt_toks) if prompt_toks is not None else None,
            "usage_reasoning_tokens": int(reasoning_toks) if reasoning_toks is not None else None,
            "usage_completion_tokens": int(completion_toks) if completion_toks is not None else None,
            "latency_ms": float(latency_ms),
        }
        return row
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000.0
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "reasoning": json.dumps(reasoning_cfg) if reasoning_cfg else None,
            "query_id": qid,
            "query_text": qtext,
            "k": int(k),
            "retrieved": json.dumps([]),
            "response_json": None,
            "response_text": None,
            "selected_index": None,
            "selected_doc_id": None,
            "parse_error": f"request_error: {e}",
            "any_relevant_in_topk": None,
            "correct": None,
            "usage_prompt_tokens": None,
            "usage_reasoning_tokens": None,
            "usage_completion_tokens": None,
            "latency_ms": float(latency_ms),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SciDocs source DB (from download script)")
    ap.add_argument("--results_db", required=True, help="Output DuckDB for results")
    ap.add_argument("--index_dir", required=True, help="Dir with faiss.index + meta.jsonl")
    ap.add_argument("--emb", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--prompt_path", default="prompts/select_doc/prompt.md.jinja")
    ap.add_argument("--model", default="openai/gpt-5")
    ap.add_argument("--use_reranker", action="store_true")
    ap.add_argument("--reranker_name", default="BAAI/bge-reranker-base")
    ap.add_argument("--snippet_len", "--snippet-len", type=int, default=220, dest="snippet_len")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--run_id", default=None, help="Tag to identify this run; default: model+timestamp")
    ap.add_argument("--sample", type=int, default=None,
                help="If set, only run this many queries from scidocs_queries (useful for quick tests).")

    # Reasoning controls (OpenRouter unified)   [oai_citation:6‡OpenRouter](https://openrouter.ai/docs/use-cases/reasoning-tokens)
    ap.add_argument("--reasoning_effort", choices=["low","medium","high"], default=None)
    ap.add_argument("--reasoning_max_tokens", type=int, default=None)
    ap.add_argument("--reasoning_enabled", action="store_true")
    ap.add_argument("--reasoning_exclude", action="store_true")

    args = ap.parse_args()

    # Load prompt
    tmpl_text = Path(args.prompt_path).read_text()
    prompt_tmpl = Template(tmpl_text)

    # OpenRouter client
    client = openrouter_client()

    # Open FAISS + embedder
    index_dir = Path(args.index_dir)
    idx_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.jsonl"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing {idx_path} or {meta_path}")
    index = faiss.read_index(str(idx_path))
    meta  = load_meta(str(meta_path))
    index_lock = Lock()  # serialize .search if multiple threads share the same index

    embedder = SentenceTransformer(args.emb)

    # Source data
    # Source data
    src = duckdb.connect(args.db, read_only=True)
    queries = fetch_scidocs_queries(src)
    if args.sample:
        queries = queries[:args.sample]
        print(f"⚙️ Sampling {len(queries)} of {len(fetch_scidocs_queries(src))} total queries")
    qrels_map = fetch_qrels_map(src)              # query_id -> {doc_id: rel}

    # Output DB
    out = duckdb.connect(args.results_db)
    ensure_results_table(out, args.results_db)

    run_id = args.run_id or f"{args.model}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    reasoning_cfg = build_reasoning_config(args)

    # Worker wrapper (so we can persist rows as we go)
    def submit(qitem):
        row = process_query(
            qitem=qitem,
            model_name=args.model,
            client=client,
            index=index,
            index_lock=index_lock,
            meta=meta,
            embedder=embedder,
            con=src,
            qrels_map=qrels_map,
            k=args.k,
            prompt_tmpl=prompt_tmpl,
            snippet_len=args.snippet_len,
            use_reranker=args.use_reranker,
            reranker_name=args.reranker_name,
            reasoning_cfg=reasoning_cfg,
        )
        # insert row
        out.execute("""
          INSERT INTO scidocs_rag_results
          SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        """, [
            run_id,
            row["ts"],
            row["model"],
            row["reasoning"],
            row["query_id"],
            row["query_text"],
            row["k"],
            row["retrieved"],
            row["response_json"],
            row["response_text"],
            row["selected_index"],
            row["selected_doc_id"],
            row["parse_error"],
            row["any_relevant_in_topk"],
            row["correct"],
            row["usage_prompt_tokens"],
            row["usage_reasoning_tokens"],
            row["usage_completion_tokens"],
            row["latency_ms"],
        ])
        return row

    # Progress bar with threads
    pbar = tqdm(total=len(queries), desc=f"RAG eval ({args.model})")
    if args.threads > 1:
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futs = [ex.submit(submit, q) for q in queries]
            for _ in as_completed(futs):
                pbar.update(1)
    else:
        for q in queries:
            submit(q)
            pbar.update(1)
    pbar.close()

    src.close()
    out.close()
    print(f"✅ Completed run_id={run_id} into {args.results_db}")

if __name__ == "__main__":
    main()
