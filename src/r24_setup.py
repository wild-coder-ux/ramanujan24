"""
r24_setup.py — Ramanujan-24 Smart Setup
========================================
Run this ONCE on any new document before using r24_query.py.

What it does:
  1. Loads your PDF (or SQuAD)
  2. Detects document type (structured / technical / poetic / factual)
  3. Embeds corpus (or loads cache)
  4. Sweeps Stage0_K at 20/40/60/70/80% coverage
  5. Finds the best K for speed-optimised and accuracy-optimised modes
  6. Saves r24_config.json — all other scripts read this automatically

Usage:
  python r24_setup.py --pdf my_document.pdf
  python r24_setup.py --pdf my_document.pdf --out my_config.json
  python r24_setup.py --squad                  # use SQuAD instead

Output: r24_config.json (or --out path)
"""

import numpy as np
import ctypes, os, re, time, json, argparse
import torch
from sentence_transformers import SentenceTransformer

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME  = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE  = 4
MAX_QUERIES = 300      # enough for a reliable sweep
STAGE1_DIMS = 24
STAGE2_DIMS = 96
STAGE1_K    = 200
STAGE2_K    = 50
STAGE3_K    = 10
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

# ── Document type detection ───────────────────────────────────────────────────

STRUCTURED_KEYWORDS  = ["theorem","definition","algorithm","equation","figure",
                         "chapter","section","proof","example","exercise","table"]
POETIC_KEYWORDS      = ["verse","mantra","hymn","stanza","siva","shiva","lord",
                         "soul","liberation","divine","sacred","chant","guru"]
FACTUAL_KEYWORDS     = ["born","died","located","founded","known","called",
                         "population","according","however","although","whereas"]

def detect_doc_type(passages):
    """
    Classify document into one of three types by vocabulary fingerprint.
    Returns: ('structured' | 'poetic' | 'factual', confidence_dict)
    """
    sample = " ".join(passages[:200]).lower()
    words  = sample.split()
    total  = max(len(words), 1)

    scores = {
        "structured" : sum(sample.count(k) for k in STRUCTURED_KEYWORDS) / total,
        "poetic"     : sum(sample.count(k) for k in POETIC_KEYWORDS)     / total,
        "factual"    : sum(sample.count(k) for k in FACTUAL_KEYWORDS)    / total,
    }

    # Sentence-length heuristic: poetic texts have shorter sentences
    sentences = re.split(r'[.!?]', " ".join(passages[:50]))
    avg_sent_len = np.mean([len(s.split()) for s in sentences if s.strip()])

    if avg_sent_len < 12:
        scores["poetic"] *= 2.0   # boost poetic if short sentences

    doc_type = max(scores, key=scores.get)

    # Fallback: if all scores are tiny, use sentence length
    if max(scores.values()) < 0.001:
        doc_type = "poetic" if avg_sent_len < 12 else "factual"

    return doc_type, scores, avg_sent_len

def doc_type_advice(doc_type):
    advice = {
        "structured": (
            "Technical/structured document detected.\n"
            "  Topic sentences carry most signal — Pingala performs well.\n"
            "  Expected: Pingala retention 85-92%, 10-15x speedup vs BM25."
        ),
        "poetic": (
            "Poetic/spiritual document detected.\n"
            "  Repeated vocabulary reduces BM25 advantage.\n"
            "  Expected: Pingala retention ~90%, matches BM25, 15-20x faster."
        ),
        "factual": (
            "Factual/encyclopaedic document detected.\n"
            "  Keyword-rich passages favour BM25 over Pingala.\n"
            "  Expected: Pingala retention 80-85%, BM25 leads by ~13pp.\n"
            "  Consider hybrid mode (Pingala ∪ BM25) for best results."
        ),
    }
    return advice.get(doc_type, "")

# ── Loaders ───────────────────────────────────────────────────────────────────

def is_garbage(t):
    garbage = sum(1 for c in t if ord(c) > 127 or c in '@®°_$')
    return garbage / max(len(t), 1) > 0.05

def is_noise(t):
    if re.search(r'\b\d{3,}\b.*\b\d{3,}\b', t): return True
    if t.startswith('•') or t.startswith('·'):   return True
    if len(t.split()) < 10:                       return True
    tamil_chars = sum(1 for c in t if '\u0B80' <= c <= '\u0BFF')
    if tamil_chars / max(len(t), 1) > 0.3:       return True
    if is_garbage(t):                             return True
    return False

def load_pdf(pdf_path):
    import fitz
    print(f"  📄 Reading {pdf_path}…")
    doc, passages, buffer = fitz.open(pdf_path), [], ""
    for page in doc:
        for block in page.get_text("blocks"):
            if block[6] != 0: continue
            t = block[4].strip().replace("\n", " ")
            tamil_chars = sum(1 for c in t if '\u0B80' <= c <= '\u0BFF')
            if tamil_chars / max(len(t), 1) > 0.3 or is_garbage(t):
                if buffer and len(buffer.split()) >= 10 and not is_garbage(buffer):
                    passages.append(buffer)
                buffer = ""
                continue
            if is_noise(t): continue
            if buffer:
                if not buffer.rstrip().endswith(('.', '?', '!', '"', "'", '~')):
                    buffer = buffer.rstrip() + " " + t
                    continue
                else:
                    if len(buffer.split()) >= 10: passages.append(buffer)
                    buffer = t
            else:
                buffer = t
    if buffer and len(buffer.split()) >= 10 and not is_garbage(buffer):
        passages.append(buffer)
    return passages

def load_squad():
    from datasets import load_dataset
    print("  📄 Loading SQuAD validation set…")
    ds = load_dataset("squad", split="validation")
    seen, passages, passage_idx = set(), [], {}
    for row in ds:
        ctx = row["context"]
        if ctx not in seen:
            seen.add(ctx)
            passage_idx[ctx] = len(passages)
            passages.append(ctx)
    return passages

def make_queries(passages, n=MAX_QUERIES):
    """Build self-queries from first sentence of each passage."""
    queries, gold_ids = [], []
    for i, p in enumerate(passages):
        first = p.split(".")[0].strip()
        if len(first) > 20:
            queries.append(first)
            gold_ids.append(i)
        if len(queries) >= n:
            break
    return queries, gold_ids

# ── Hash ──────────────────────────────────────────────────────────────────────

def load_lib(so_name, fn_name):
    path = os.path.abspath(so_name)
    if not os.path.exists(path): return None
    lib = ctypes.CDLL(path)
    fn  = getattr(lib, fn_name)
    fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
    fn.restype  = None
    return fn

def hash_text(fn, text):
    arr = (ctypes.c_float * 24)()
    fn(text.encode("utf-8", errors="replace"), arr)
    v = np.array(list(arr), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def build_hash_matrix(fn, passages):
    out = np.zeros((len(passages), 24), dtype=np.float32)
    for i, p in enumerate(passages):
        out[i] = hash_text(fn, p)
    return out

# ── Search ────────────────────────────────────────────────────────────────────

def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms

def cascade(q_text, q_emb, stage0_ids, embs):
    survivors0 = stage0_ids
    s1k    = min(STAGE1_K, len(survivors0))
    cands1 = normalize(embs[survivors0, :STAGE1_DIMS])
    q24    = normalize(q_emb[None, :STAGE1_DIMS])
    surv1  = survivors0[np.argsort((cands1 @ q24.T).squeeze())[::-1][:s1k]]

    s2k    = min(STAGE2_K, len(surv1))
    cands2 = normalize(embs[surv1, :STAGE2_DIMS])
    q96    = normalize(q_emb[None, :STAGE2_DIMS])
    surv2  = surv1[np.argsort((cands2 @ q96.T).squeeze())[::-1][:s2k]]

    cands3 = normalize(embs[surv2])
    q768   = normalize(q_emb[None, :])
    scores = (cands3 @ q768.T).squeeze()
    if scores.ndim == 0: scores = scores[None]
    top    = np.argsort(scores)[::-1][:STAGE3_K]
    return surv2[top]

def brute_force(q_emb, embs):
    scores = (normalize(embs) @ normalize(q_emb[None, :]).T).squeeze()
    return np.argsort(scores)[::-1][:STAGE3_K]

def run_sweep(k, hash_mat, pingala_fn, q_texts, q_embs, gold_ids, embs):
    hits = 0
    k    = min(k, len(embs))
    for q_text, q_emb, gold in zip(q_texts, q_embs, gold_ids):
        q_hash  = hash_text(pingala_fn, q_text)
        stage0  = np.argsort(hash_mat @ q_hash)[::-1][:k]
        if gold in cascade(q_text, q_emb, stage0, embs):
            hits += 1
    return hits / len(gold_ids) * 100

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ramanujan-24 Smart Setup")
    parser.add_argument("--pdf",   default=None,             help="Path to PDF file")
    parser.add_argument("--squad", action="store_true",      help="Use SQuAD instead of PDF")
    parser.add_argument("--cache", default=None,             help="Path to embeddings cache (.npy)")
    parser.add_argument("--out",   default="r24_config.json",help="Output config file path")
    args = parser.parse_args()

    if not args.pdf and not args.squad:
        print("❌ Provide --pdf <file.pdf> or --squad")
        return

    print("\n" + "=" * 62)
    print("  Ramanujan-24 — Smart Setup")
    print("=" * 62)

    # ── Load passages ─────────────────────────────────────────────
    if args.squad:
        passages  = load_squad()
        doc_label = "SQuAD"
        cache_path = args.cache or "squad_vectors.npy"
    else:
        passages  = load_pdf(args.pdf)
        doc_label = os.path.basename(args.pdf)
        stem      = os.path.splitext(doc_label)[0].lower().replace(" ", "_")
        cache_path = args.cache or f"{stem}_vectors.npy"

    N = len(passages)
    print(f"\n  ✅ {N:,} passages loaded from {doc_label}")

    # ── Detect document type ──────────────────────────────────────
    doc_type, scores, avg_sent = detect_doc_type(passages)
    print(f"\n  🔍 Document type  : {doc_type.upper()}")
    print(f"     Avg sentence   : {avg_sent:.1f} words")
    print(f"     Type scores    : structured={scores['structured']:.4f} "
          f"poetic={scores['poetic']:.4f} factual={scores['factual']:.4f}")
    print(f"\n  💡 {doc_type_advice(doc_type)}")

    # ── Load hash lib ─────────────────────────────────────────────
    pingala_fn = load_lib("libr24_pingala.so", "hash24_pingala")
    if not pingala_fn:
        print("\n❌ libr24_pingala.so not found.")
        print("   Compile: gcc -O3 -shared -fPIC r24_pingala.c -o libr24_pingala.so")
        return

    # ── Load or build embeddings ──────────────────────────────────
    print(f"\n  🤖 Loading model ({DEVICE.upper()})…")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)

    if os.path.exists(cache_path):
        embs = np.load(cache_path)
        if len(embs) == N:
            print(f"  📦 Loaded cached embeddings ({N:,} passages)")
        else:
            print(f"  ⚠️  Cache mismatch — re-embedding…")
            embs = None
    else:
        embs = None

    if embs is None:
        if DEVICE == "cuda": torch.cuda.empty_cache()
        print(f"  🔢 Embedding {N:,} passages…")
        embs = model.encode(passages, batch_size=BATCH_SIZE,
                            show_progress_bar=True,
                            normalize_embeddings=True, convert_to_numpy=True)
        np.save(cache_path, embs)
        print(f"  💾 Saved to {cache_path}")

    # ── Build hash matrix & queries ───────────────────────────────
    print(f"\n  ⚡ Building Pingala hash matrix…")
    hash_mat = build_hash_matrix(pingala_fn, passages)

    queries, gold_ids = make_queries(passages, MAX_QUERIES)
    print(f"  🔍 Embedding {len(queries)} sweep queries…")
    if DEVICE == "cuda": torch.cuda.empty_cache()
    q_embs = model.encode(queries, batch_size=BATCH_SIZE,
                          show_progress_bar=False,
                          normalize_embeddings=True, convert_to_numpy=True)

    # Brute-force ceiling
    bf_hits = sum(
        1 for q_emb, gold in zip(q_embs, gold_ids)
        if gold in brute_force(q_emb, embs)
    )
    bf_ret = bf_hits / len(gold_ids) * 100
    print(f"\n  📊 Brute-force ceiling: {bf_ret:.1f}%")

    # ── K sweep ───────────────────────────────────────────────────
    coverages   = [0.20, 0.40, 0.60, 0.70, 0.80, 0.90]
    k_values    = [max(10, int(c * N)) for c in coverages]
    k_values    = sorted(set(k_values))   # deduplicate

    print(f"\n  🔁 Sweeping Stage0_K across {len(k_values)} values…\n")
    print(f"  {'K':>6}  {'Coverage':>9}  {'Retention':>10}  {'vs ceiling':>11}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*10}  {'─'*11}")

    sweep_results = []
    for k in k_values:
        ret = run_sweep(k, hash_mat, pingala_fn, queries, q_embs, gold_ids, embs)
        cov = k / N * 100
        vs  = ret / bf_ret * 100
        sweep_results.append({"k": k, "coverage": round(cov,1),
                               "retention": round(ret,1), "vs_ceiling": round(vs,1)})
        print(f"  {k:>6}  {cov:>8.1f}%  {ret:>9.1f}%  {vs:>10.1f}%")

    # ── Find operating points ─────────────────────────────────────
    # Speed-optimised: highest K where retention gain per +1% coverage < 0.3pp
    # Accuracy-optimised: first K where retention > 85% of ceiling
    acc_target  = 0.85 * bf_ret
    speed_k     = k_values[0]
    accuracy_k  = k_values[-1]

    for i, r in enumerate(sweep_results):
        if r["retention"] >= acc_target:
            accuracy_k = r["k"]
            break

    # Find speed knee: biggest retention gain per unit K
    gains = []
    for i in range(1, len(sweep_results)):
        dk  = sweep_results[i]["k"] - sweep_results[i-1]["k"]
        dr  = sweep_results[i]["retention"] - sweep_results[i-1]["retention"]
        gains.append(dr / max(dk, 1))

    if gains:
        knee_i  = gains.index(min(gains))   # where gain flattens most
        speed_k = sweep_results[max(0, knee_i)]["k"]

    speed_ret   = next(r["retention"] for r in sweep_results if r["k"] == speed_k)
    accuracy_ret = next(r["retention"] for r in sweep_results if r["k"] == accuracy_k)

    # ── Print recommendation ──────────────────────────────────────
    print(f"\n  {'='*58}")
    print(f"  RECOMMENDED OPERATING POINTS")
    print(f"  {'='*58}")
    print(f"  Speed-optimised   K={speed_k:,}  "
          f"coverage={speed_k/N*100:.0f}%  retention={speed_ret:.1f}%")
    print(f"  Accuracy-optimised K={accuracy_k:,}  "
          f"coverage={accuracy_k/N*100:.0f}%  retention={accuracy_ret:.1f}%")
    print(f"  {'='*58}")

    # ── Save config ───────────────────────────────────────────────
    config = {
        "document"       : doc_label,
        "cache_path"     : cache_path,
        "pdf_path"       : args.pdf,
        "num_passages"   : N,
        "doc_type"       : doc_type,
        "doc_type_scores": {k: round(v, 5) for k, v in scores.items()},
        "avg_sentence_len": round(float(avg_sent), 1),
        "model_name"     : MODEL_NAME,
        "brute_force_ceiling": round(bf_ret, 1),
        "sweep"          : sweep_results,
        "operating_points": {
            "speed": {
                "stage0_k" : speed_k,
                "coverage" : round(speed_k / N * 100, 1),
                "retention": speed_ret
            },
            "accuracy": {
                "stage0_k" : accuracy_k,
                "coverage" : round(accuracy_k / N * 100, 1),
                "retention": accuracy_ret
            }
        },
        "pipeline": {
            "stage1_dims": STAGE1_DIMS,
            "stage2_dims": STAGE2_DIMS,
            "stage1_k"   : STAGE1_K,
            "stage2_k"   : STAGE2_K,
            "stage3_k"   : STAGE3_K,
            "batch_size" : BATCH_SIZE,
            "device"     : DEVICE
        }
    }

    with open(args.out, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  ✅ Config saved to {args.out}")
    print(f"\n  Next step — run the query tool:")
    print(f"    python r24_query.py --config {args.out} --mode speed")
    print(f"    python r24_query.py --config {args.out} --mode accuracy")
    print()

if __name__ == "__main__":
    main()
