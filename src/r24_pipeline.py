"""
r24_pipeline.py — Ramanujan-24 Full Pipeline
Tests TWO Stage 0 strategies back to back:
  A) Plain C hash (original)
  B) Pingala hash (Fibonacci-weighted, our innovation)

Then neural cascade:
  Stage 1: 24-dim  → 200
  Stage 2: 96-dim  → 50
  Stage 3: 768-dim → 10

Works on:
  - SQuAD (set MODE = "squad")
  - PDF document (set MODE = "pdf", set PDF_PATH)

Run: python r24_pipeline.py
"""

import numpy as np
import ctypes, os, time, re
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "nomic-ai/nomic-embed-text-v1.5"

MODE        = "pdf"
PDF_PATH    = "d2l_deep_learning.pdf"
DATA_PATH   = "d2l_deep_learning_vectors.npy"

STAGE0_K    = 4000
STAGE1_DIMS = 24
STAGE2_DIMS = 96
STAGE3_DIMS = 768
STAGE1_K    = 200
STAGE2_K    = 50
STAGE3_K    = 10
MAX_QUERIES = 500
BATCH_SIZE  = 4      # safe for 8GB VRAM; drop to 2 if OOM

# ── Device selection ─────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────

def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms

# ── Hash loaders ──────────────────────────────────────────────────────────────

def load_lib(so_name, fn_name):
    path = os.path.abspath(so_name)
    if not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    fn  = getattr(lib, fn_name)
    fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
    fn.restype  = None
    return fn

def hash_text(fn, text):
    arr = (ctypes.c_float * 24)()
    fn(text.encode("utf-8", errors="replace"), arr)
    v = np.array(list(arr), dtype=np.float32)
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def build_hash_matrix(fn, passages, label):
    print(f"⚡ Building {label} vectors for {len(passages):,} passages…")
    t0  = time.perf_counter()
    out = np.zeros((len(passages), 24), dtype=np.float32)
    for i, p in enumerate(passages):
        out[i] = hash_text(fn, p)
    elapsed = time.perf_counter() - t0
    print(f"   Done in {elapsed:.3f}s")
    return out

# ── Search ────────────────────────────────────────────────────────────────────

def cascade(q_text, q_emb, hash_mat, hash_fn, embs):
    N        = len(embs)
    stage0_k = min(STAGE0_K, N)

    q_hash     = hash_text(hash_fn, q_text)
    scores0    = hash_mat @ q_hash
    survivors0 = np.argsort(scores0)[::-1][:stage0_k]

    s1k        = min(STAGE1_K, len(survivors0))
    cands1     = normalize(embs[survivors0, :STAGE1_DIMS])
    q24        = normalize(q_emb[None, :STAGE1_DIMS])
    survivors1 = survivors0[np.argsort((cands1 @ q24.T).squeeze())[::-1][:s1k]]

    s2k        = min(STAGE2_K, len(survivors1))
    cands2     = normalize(embs[survivors1, :STAGE2_DIMS])
    q96        = normalize(q_emb[None, :STAGE2_DIMS])
    survivors2 = survivors1[np.argsort((cands2 @ q96.T).squeeze())[::-1][:s2k]]

    cands3     = normalize(embs[survivors2])
    q768       = normalize(q_emb[None, :])
    top10      = np.argsort((cands3 @ q768.T).squeeze())[::-1][:STAGE3_K]
    return survivors2[top10]

def brute_force(q_emb, embs):
    scores = (normalize(embs) @ normalize(q_emb[None, :]).T).squeeze()
    return np.argsort(scores)[::-1][:STAGE3_K]

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_squad():
    from datasets import load_dataset
    ds = load_dataset("squad", split="validation")
    seen, passages, passage_idx = set(), [], {}
    for row in ds:
        ctx = row["context"]
        if ctx not in seen:
            seen.add(ctx)
            passage_idx[ctx] = len(passages)
            passages.append(ctx)
    queries, gold_ids = [], []
    for row in ds:
        if row["context"] not in passage_idx:
            continue
        queries.append(row["question"])
        gold_ids.append(passage_idx[row["context"]])
        if len(queries) >= MAX_QUERIES:
            break
    return passages, queries, gold_ids

def is_noise(t):
    """Filter out TOC lines, captions, and other low-signal blocks."""
    if re.search(r'\b\d{3,}\b.*\b\d{3,}\b', t):  # two 3-digit numbers = page refs / TOC
        return True
    if t.startswith('•') or t.startswith('·'):     # bullet lists
        return True
    if len(t.split()) < 10:                         # too short to be a real passage
        return True
    return False

def load_pdf(pdf_path):
    try:
        import fitz
    except ImportError:
        raise ImportError("Run: pip install pymupdf")

    print(f"📄 Reading {pdf_path}…")
    doc = fitz.open(pdf_path)
    passages = []
    for page in doc:
        for block in page.get_text("blocks"):
            if block[6] == 0:                              # text block only
                t = block[4].strip().replace("\n", " ")
                if len(t) > 80 and not is_noise(t):
                    passages.append(t)

    queries, gold_ids = [], []
    for i, p in enumerate(passages):
        first = p.split(".")[0].strip()
        if len(first) > 20:
            queries.append(first)
            gold_ids.append(i)
        if len(queries) >= MAX_QUERIES:
            break

    print(f"   Extracted {len(passages)} passages, {len(queries)} queries")
    return passages, queries, gold_ids

def get_or_embed(model, passages, cache_path):
    if os.path.exists(cache_path):
        print(f"📦 Loading cached embeddings from {cache_path}")
        embs = np.load(cache_path)
        if len(embs) == len(passages):
            return embs
        print("   Cache size mismatch — re-embedding…")

    print(f"🔢 Embedding {len(passages)} passages "
          f"({'GPU' if DEVICE == 'cuda' else 'CPU'}, batch={BATCH_SIZE})…")

    # Free stale VRAM before starting
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    embs = model.encode(
        passages,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Save cache — handle flat DATA_PATH (no subdirectory)
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, embs)
    print(f"💾 Saved to {cache_path}")
    return embs

# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(label, hash_fn, hash_mat, q_texts, q_embs, gold_ids, embs):
    hits, times = 0, []
    for q_text, q_emb, gold in zip(q_texts, q_embs, gold_ids):
        t0 = time.perf_counter()
        if gold in cascade(q_text, q_emb, hash_mat, hash_fn, embs):
            hits += 1
        times.append(time.perf_counter() - t0)
    return hits, np.mean(times) * 1000

def main():
    print("=" * 64)
    print("  Ramanujan-24 Pipeline — Plain Hash vs Pingala Hash")
    print(f"  Mode: {MODE.upper()} | Device: {DEVICE.upper()}")
    print("=" * 64)

    plain_fn   = load_lib("libr24.so",        "hash24")
    pingala_fn = load_lib("libr24_pingala.so", "hash24_pingala")

    if plain_fn   is None: print("⚠️  libr24.so not found")
    if pingala_fn is None: print("⚠️  libr24_pingala.so not found — compile r24_pingala.c")

    if MODE == "squad":
        passages, queries, gold_ids = load_squad()
        cache_path = DATA_PATH
    else:
        passages, queries, gold_ids = load_pdf(PDF_PATH)
        cache_path = DATA_PATH   # always use explicit DATA_PATH

    print(f"\n   Passages : {len(passages):,}")
    print(f"   Queries  : {len(queries)}")

    print(f"\n🤖 Loading model ({DEVICE.upper()})…")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)

    embs = get_or_embed(model, passages, cache_path)

    assert len(embs) == len(passages), \
        f"Embedding size {len(embs)} != passages {len(passages)}"

    plain_mat   = build_hash_matrix(plain_fn,   passages, "plain hash")   if plain_fn   else None
    pingala_mat = build_hash_matrix(pingala_fn, passages, "pingala hash") if pingala_fn else None

    print(f"\n🔍 Embedding {len(queries)} queries…")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    q_embs = model.encode(
        queries,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("\n⚡ Running brute-force baseline…")
    bf_hits, bf_times = 0, []
    for q_emb, gold in tqdm(zip(q_embs, gold_ids), total=len(queries)):
        t0 = time.perf_counter()
        if gold in brute_force(q_emb, embs): bf_hits += 1
        bf_times.append(time.perf_counter() - t0)
    bf_ms = np.mean(bf_times) * 1000

    results = {}
    if plain_fn and plain_mat is not None:
        print("\n⚡ Running plain hash cascade…")
        hits, ms = run_benchmark("plain", plain_fn, plain_mat, queries, q_embs, gold_ids, embs)
        results["Plain hash + 24-dim"] = (hits, ms)

    if pingala_fn and pingala_mat is not None:
        print("\n⚡ Running Pingala hash cascade…")
        hits, ms = run_benchmark("pingala", pingala_fn, pingala_mat, queries, q_embs, gold_ids, embs)
        results["Pingala hash + 24-dim"] = (hits, ms)

    n = len(queries)
    print("\n" + "=" * 64)
    print(f"  RESULTS — {MODE.upper()} | {len(passages):,} passages | {n} queries")
    print("=" * 64)
    print(f"  {'Method':<28} {'Recall@10':<12} {'Retention':<12} {'ms/q':<10} {'Speedup'}")
    print(f"  {'-'*27:<28} {'-'*10:<12} {'-'*10:<12} {'-'*8:<10} {'-'*7}")
    print(f"  {'Brute-force (768-dim)':<28} {bf_hits/n*100:<12.1f} {'100.0':<12} {bf_ms:<10.2f} 1.0x")

    for label, (hits, ms) in results.items():
        retention = hits / max(bf_hits, 1) * 100
        speedup   = bf_ms / ms
        marker    = " ✅" if retention >= 85 else " ⚠️"
        print(f"  {label:<28} {hits/n*100:<12.1f} {retention:<12.1f} {ms:<10.2f} {speedup:.1f}x{marker}")

    print("=" * 64)
    print(f"\n  Stage 0 keeps: {STAGE0_K} of {len(passages):,} passages ({STAGE0_K/len(passages)*100:.1f}%)")
    print(f"  Key question: does Pingala beat plain hash on {MODE.upper()}?")

if __name__ == "__main__":
    main()
