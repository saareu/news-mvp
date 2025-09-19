# embed_index_pipeline.py
# Python 3.11
# pip install polars duckdb sentence-transformers faiss-cpu torch --extra-index-url https://download.pytorch.org/whl/cpu
## python scripts/embeddings.py --parquet_glob "data/parquet/*/*/*.parquet" --lang he
import json
import argparse
import pathlib
import numpy as np
import polars as pl
import duckdb
import faiss
from sentence_transformers import SentenceTransformer
import torch
import glob

ART = pathlib.Path("artifacts")
ART.mkdir(exist_ok=True)
IDX_PATH = ART / "index.faiss"
IDMAP_PATH = ART / "idmap.txt"
MMAP_PATH = ART / "embeddings.mmap"
META_PATH = ART / "embeddings.meta.json"
MANIFEST_DB = ART / "manifest.duckdb"


# -------- 1) Parquet -> streaming batches  --------
def iter_batches(
    parquet_glob: str,
    batch_size: int = 2048,
    min_len: int = 10,
    lang: str | None = None,
):
    scan = pl.scan_parquet(parquet_glob).select(
        ["article_id", "description", "language"]
    )
    if lang:
        scan = scan.filter(pl.col("language") == lang)
    scan = scan.filter(pl.col("description").str.len_chars() >= min_len)
    ids, txts = [], []
    row_count = 0
    df = scan.collect()
    for row in df.iter_rows(named=True):
        ids.append(row["article_id"])
        txts.append(row["description"])
        row_count += 1
        if len(ids) >= batch_size:
            print(f"[debug] Yielding batch of {len(ids)} (row {row_count})")
            yield ids, txts
            ids, txts = [], []
    if ids:
        print(f"[debug] Yielding final batch of {len(ids)} (row {row_count})")
        yield ids, txts


# -------- 2) Model loader + encoding  --------
def load_sbert(name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(name, device=device)
    dim = model.get_sentence_embedding_dimension()
    return model, dim


@torch.inference_mode()
def encode_norm(model, texts: list[str], batch_size: int = 128) -> np.ndarray:
    X = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    X = X.astype("float32")
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12  # cosine-ready
    return X


# -------- 3) FAISS index + idmap + memmap state  --------
def load_or_init(dim: int):
    if IDX_PATH.exists():
        index = faiss.read_index(str(IDX_PATH))
        ids = (
            IDMAP_PATH.read_text(encoding="utf-8").splitlines()
            if IDMAP_PATH.exists()
            else []
        )
        meta = (
            json.loads(META_PATH.read_text())
            if META_PATH.exists()
            else {"rows": 0, "dim": dim}
        )
        mmap = None
        if MMAP_PATH.exists() and meta["rows"] > 0:
            mmap = np.memmap(
                MMAP_PATH, dtype="float32", mode="r+", shape=(meta["rows"], meta["dim"])
            )
        return index, ids, mmap, meta
    index = faiss.IndexFlatIP(dim)
    ids, meta, mmap = [], {"rows": 0, "dim": dim}, None
    META_PATH.write_text(json.dumps(meta))
    return index, ids, mmap, meta


def save_state(index, ids, meta):
    faiss.write_index(index, str(IDX_PATH))
    IDMAP_PATH.write_text("\n".join(ids), encoding="utf-8")
    META_PATH.write_text(json.dumps(meta))


# -------- 4) Manifest of embedded IDs (avoids duplicates) --------
def init_manifest():
    con = duckdb.connect(str(MANIFEST_DB))
    con.execute("CREATE TABLE IF NOT EXISTS embedded(article_id VARCHAR PRIMARY KEY);")
    return con


def unseen_idx(con, ids: list[str]) -> list[int]:
    if not ids:
        return []
    # Use a parameterized VALUES table to anti-join quickly
    vals = ",".join(["(?)"] * len(ids))
    q = f"""
WITH v(article_id) AS (VALUES {vals})
SELECT row_number() OVER ()-1 AS pos, article_id
FROM v
LEFT JOIN embedded USING (article_id)
WHERE embedded.article_id IS NULL;
"""
    res = con.execute(q, ids).fetchall()
    return [pos for (pos, _) in res]


def register_ids(con, ids: list[str]):
    if ids:
        con.executemany(
            "INSERT OR IGNORE INTO embedded(article_id) VALUES (?)", [(i,) for i in ids]
        )


# -------- 5) Append to memmap safely (resize by replace) --------
def append_vectors(mmap, meta, vecs: np.ndarray):
    if vecs.size == 0:
        return mmap
    dim = meta["dim"]
    assert vecs.shape[1] == dim
    start, end = meta["rows"], meta["rows"] + vecs.shape[0]
    new_path = ART / "embeddings.tmp.mmap"
    new_mmap = np.memmap(new_path, dtype="float32", mode="w+", shape=(end, dim))
    if mmap is not None and start > 0:
        chunk = 1_000_000 // max(1, dim)  # ~1M floats per chunk
        for s in range(0, start, chunk):
            e = min(start, s + chunk)
            new_mmap[s:e, :] = mmap[s:e, :]
        # --- Explicitly close the old mmap ---
        if hasattr(mmap, "_mmap"):
            mmap._mmap.close()
        del mmap
    new_mmap[start:end, :] = vecs
    new_mmap.flush()
    # --- Explicitly close the new mmap before renaming ---
    if hasattr(new_mmap, "_mmap"):
        new_mmap._mmap.close()
    del new_mmap
    new_path.replace(MMAP_PATH)
    meta["rows"] = end
    return np.memmap(MMAP_PATH, dtype="float32", mode="r+", shape=(meta["rows"], dim))


# -------- 6) Main pipeline: embed + index incrementally --------
def embed_and_index(parquet_glob: str, lang="he", batch_docs=2048, sbert_bs=128):
    model, dim = load_sbert()
    index, idmap, mmap, meta = load_or_init(dim)
    con = init_manifest()

    total_new = 0
    batch_num = 0
    for ids, texts in iter_batches(parquet_glob, batch_size=batch_docs, lang=lang):
        batch_num += 1
        print(f"[debug] Processing batch {batch_num} with {len(ids)} ids")
        keep = unseen_idx(con, ids)
        print(f"[debug] {len(keep)} unseen ids in batch {batch_num}")
        if not keep:
            continue
        ids_new = [ids[i] for i in keep]
        vecs = encode_norm(model, [texts[i] for i in keep], batch_size=sbert_bs)
        print(f"[debug] Adding {vecs.shape[0]} vectors to index and mmap")
        mmap = append_vectors(mmap, meta, vecs)
        idmap.extend(ids_new)
        index.add(vecs)
        register_ids(con, ids_new)
        total_new += len(ids_new)

    print(
        f"[debug] Finished embedding. Total new: {total_new}, total rows: {meta['rows']}"
    )
    save_state(index, idmap, meta)
    return {"added": total_new, "rows": meta["rows"], "dim": meta["dim"]}


# -------- 7) Search helpers --------
def neighbors_by_text(
    query: str,
    k=10,
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
):
    model, _ = load_sbert(model_name)
    qv = encode_norm(model, [query])
    index = faiss.read_index(str(IDX_PATH))
    D, indices = index.search(qv, k)
    ids = IDMAP_PATH.read_text(encoding="utf-8").splitlines()
    return [(ids[i], float(d)) for i, d in zip(indices[0], D[0])]


def neighbors_by_id(query_id: str, k=10):
    # Uses stored vector if memmap exists; else raises
    if not (MMAP_PATH.exists() and META_PATH.exists()):
        raise RuntimeError(
            "No memmap embeddings available; re-embed text or store vectors."
        )
    meta = json.loads(META_PATH.read_text())
    ids = IDMAP_PATH.read_text(encoding="utf-8").splitlines()
    try:
        pos = ids.index(query_id)
    except ValueError:
        raise KeyError(f"id not found: {query_id}")
    mmap = np.memmap(
        MMAP_PATH, dtype="float32", mode="r", shape=(meta["rows"], meta["dim"])
    )
    qv = mmap[pos : pos + 1]
    index = faiss.read_index(str(IDX_PATH))
    D, indices = index.search(qv, k)
    return [(ids[i], float(d)) for i, d in zip(indices[0], D[0])]


# -------- CLI --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_glob", default="data/parquet/**/**/*.parquet")
    ap.add_argument("--lang", default="he")
    ap.add_argument("--batch_docs", type=int, default=2048)
    ap.add_argument("--sbert_bs", type=int, default=128)
    ap.add_argument("--search_text", type=str, default=None)
    ap.add_argument("--search_id", type=str, default=None)
    args = ap.parse_args()

    if args.search_text or args.search_id:
        if args.search_text:
            print(neighbors_by_text(args.search_text, k=10))
        else:
            print(neighbors_by_id(args.search_id, k=10))
        return

    stats = embed_and_index(
        args.parquet_glob, args.lang, args.batch_docs, args.sbert_bs
    )
    print({"status": "ok", **stats})


if __name__ == "__main__":
    main()

print(glob.glob("data/parquet/date=*/source=*/**/*.parquet", recursive=True))
