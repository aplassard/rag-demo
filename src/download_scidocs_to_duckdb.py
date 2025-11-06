#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download BEIR/SciDocs via ir_datasets and store into DuckDB tables:
  - scidocs_docs(doc_id TEXT, title TEXT, text TEXT, authors TEXT[],
                 year INTEGER, cited_by TEXT[], reference_ids TEXT[])
  - scidocs_queries(query_id TEXT, text TEXT, authors TEXT[],
                    year INTEGER, cited_by TEXT[], reference_ids TEXT[])
  - scidocs_qrels(query_id TEXT, doc_id TEXT, relevance INTEGER, iteration TEXT)

Usage:
  python download_scidocs_to_duckdb.py --db data/scidocs.duckdb
"""
import argparse
import duckdb
import ir_datasets
from tqdm import tqdm

def main(db_path: str):
    ds = ir_datasets.load("beir/scidocs")

    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS scidocs_docs(
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            text TEXT,
            authors TEXT[],
            year INTEGER,
            cited_by TEXT[],
            reference_ids TEXT[]
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS scidocs_queries(
            query_id TEXT PRIMARY KEY,
            text TEXT,
            authors TEXT[],
            year INTEGER,
            cited_by TEXT[],
            reference_ids TEXT[]
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS scidocs_qrels(
            query_id TEXT,
            doc_id TEXT,
            relevance INTEGER,
            iteration TEXT
        );
    """)

    con.execute("BEGIN TRANSACTION;")
    try:
        # Insert docs
        docs_it = ds.docs_iter()
        batch = []
        BATCH_SIZE = 5000
        for d in tqdm(docs_it, desc="docs"):
            batch.append((
                d.doc_id,
                getattr(d, "title", "") or "",
                getattr(d, "text", "") or "",
                list(getattr(d, "authors", []) or []),
                int(getattr(d, "year", 0) or 0),
                list(getattr(d, "cited_by", []) or []),
                list(getattr(d, "references", []) or []),
            ))
            if len(batch) >= BATCH_SIZE:
                con.executemany(
                    "INSERT OR REPLACE INTO scidocs_docs (doc_id, title, text, authors, year, cited_by, reference_ids) VALUES (?, ?, ?, ?, ?, ?, ?);",
                    batch
                )
                batch.clear()
        if batch:
            con.executemany(
                "INSERT OR REPLACE INTO scidocs_docs (doc_id, title, text, authors, year, cited_by, reference_ids) VALUES (?, ?, ?, ?, ?, ?, ?);",
                batch
            )
            batch.clear()

        # Insert queries
        q_it = ds.queries_iter()
        batch = []
        for q in tqdm(q_it, desc="queries"):
            batch.append((
                q.query_id,
                q.text,
                list(getattr(q, "authors", []) or []),
                int(getattr(q, "year", 0) or 0),
                list(getattr(q, "cited_by", []) or []),
                list(getattr(q, "references", []) or []),
            ))
            if len(batch) >= BATCH_SIZE:
                con.executemany(
                    "INSERT OR REPLACE INTO scidocs_queries (query_id, text, authors, year, cited_by, reference_ids) VALUES (?, ?, ?, ?, ?, ?);",
                    batch
                )
                batch.clear()
        if batch:
            con.executemany(
                "INSERT OR REPLACE INTO scidocs_queries (query_id, text, authors, year, cited_by, reference_ids) VALUES (?, ?, ?, ?, ?, ?);",
                batch
            )
            batch.clear()

        # Insert qrels
        qrels_it = ds.qrels_iter()
        batch = []
        for qr in tqdm(qrels_it, desc="qrels"):
            batch.append((qr.query_id, qr.doc_id, int(qr.relevance), getattr(qr, "iteration", "")))
            if len(batch) >= BATCH_SIZE * 4:
                con.executemany(
                    "INSERT INTO scidocs_qrels VALUES (?, ?, ?, ?);",
                    batch
                )
                batch.clear()
        if batch:
            con.executemany(
                "INSERT INTO scidocs_qrels VALUES (?, ?, ?, ?);",
                batch
            )
    except Exception:
        con.execute("ROLLBACK;")
        raise
    else:
        con.execute("COMMIT;")
    finally:
        con.close()
    print(f"✅ Loaded SciDocs into {db_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/scidocs.duckdb")
    args = ap.parse_args()
    main(args.db)
