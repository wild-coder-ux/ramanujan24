"""
r24_query.py — Interactive query tool for Ramanujan-24
Type any question, see what the pipeline retrieves from your PDF.
Shows results from both Pingala cascade and brute-force so you can
compare answers side by side.

Usage:
    python r24_query.py                         # uses config below
    python r24_query.py --pdf my_doc.pdf        # override PDF
    python r24_query.py --top 5                 # show top 5 results (default 3)

Requires: embeddings cache already built by legal_pipeline.py or r24_pipeline.py
"""

import numpy as np
import ctypes, os, re, time, argparse
import torch
from sentence_transformers import SentenceTransformer

# ── Config — edit these to match your legal_pipeline.py ──────────────────────
MODEL_NAME  = "nomic-ai/nomic-embed-text-v1.5"
PDF_PATH    = "Legal-Services-Agreement.pdf"
DATA_PATH   = "legal_agreement_vectors.npy"
STAGE0_K    = 15     # small doc — keep top 15 of ~25 passages
STAGE1_DIMS = 24
STAGE2_DIMS = 96
STAGE1_K    = 12
STAGE2_K    = 8
STAGE3_K    = 3      # show top 3 by default (override with --top)
BATCH_SIZE  = 4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms

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

def is_noise(t):
    if re.search(r'\b\d{3,}\b.*\b\d{3,}\b', t): return True
    if t.startswith('•') or t.startswith('·'):   return True
    if len(t.split()) < 10:                       return True
    return False

def load_pdf(pdf_path):
    import fitz
    doc = fitz.open(pdf_path)
    passages = []
    for page in doc:
        for block in page.get_text("blocks"):
            if block[6] == 0:
                t = block[4].strip().replace("\n", " ")
                if len(t) > 80 and not is_noise(t):
                    passages.append(t)
    return passages

def cascade(q_text, q_emb, hash_mat, hash_fn, embs, top_k):
    N  = len(embs)
    k0 = min(STAGE0_K, N)

    survivors0 = np.argsort(hash_mat @ hash_text(hash_fn, q_text))[::-1][:k0]

    s1k        = min(STAGE1_K, len(survivors0))
    cands1     = normalize(embs[survivors0, :STAGE1_DIMS])
    q24        = normalize(q_emb[None, :STAGE1_DIMS])
    survivors1 = survivors0[np.argsort((cands1 @ q24.T).squeeze())[::-1][:s1k]]

    s2k        = min(STAGE2_K, len(survivors1))
    cands2     = normalize(embs[survivors1, :STAGE2_DIMS])
    q96        = normalize(q_emb[None, :STAGE2_DIMS])
    survivors2 = survivors1[np.argsort((cands2 @ q96.T).squeeze())[::-1][:s2k]]

    cands3 = normalize(embs[survivors2])
    q768   = normalize(q_emb[None, :])
    scores = (cands3 @ q768.T).squeeze()
    # handle single-result edge case
    if scores.ndim == 0: scores = scores[None]
    top    = np.argsort(scores)[::-1][:top_k]
    return survivors2[top], scores[top]

def brute_force(q_emb, embs, top_k):
    scores = (normalize(embs) @ normalize(q_emb[None, :]).T).squeeze()
    top    = np.argsort(scores)[::-1][:top_k]
    return top, scores[top]

def fmt_passage(text, score, rank, width=90):
    preview = text[:300] + ("…" if len(text) > 300 else "")
    lines   = [f"  #{rank}  score={score:.3f}"]
    # word-wrap the preview
    words, line = preview.split(), ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append("     " + line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line: lines.append("     " + line)
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",  default=PDF_PATH)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--top",  type=int, default=STAGE3_K)
    args = parser.parse_args()

    top_k = args.top

    print("\n" + "=" * 64)
    print("  Ramanujan-24 — Interactive Query")
    print(f"  PDF  : {args.pdf}")
    print(f"  Top-K: {top_k}")
    print("=" * 64)

    # ── Load hashes ───────────────────────────────────────────────
    pingala_fn = load_lib("libr24_pingala.so", "hash24_pingala")
    plain_fn   = load_lib("libr24.so", "hash24")
    if not pingala_fn: print("⚠️  libr24_pingala.so not found"); return
    if not plain_fn:   print("⚠️  libr24.so not found"); return

    # ── Load passages ─────────────────────────────────────────────
    print(f"\n📄 Loading passages from {args.pdf}…")
    passages = load_pdf(args.pdf)
    print(f"   {len(passages)} passages loaded")

    # ── Load or build embeddings ──────────────────────────────────
    if os.path.exists(args.data):
        embs = np.load(args.data)
        if len(embs) == len(passages):
            print(f"📦 Loaded cached embeddings ({len(embs)} passages)")
        else:
            print(f"⚠️  Cache mismatch ({len(embs)} vs {len(passages)}) — re-embedding…")
            embs = None
    else:
        embs = None

    if embs is None:
        print(f"🤖 Loading model ({DEVICE.upper()})…")
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
        if DEVICE == "cuda": torch.cuda.empty_cache()
        embs = model.encode(passages, batch_size=BATCH_SIZE,
                            show_progress_bar=True,
                            normalize_embeddings=True, convert_to_numpy=True)
        np.save(args.data, embs)
        print(f"💾 Saved to {args.data}")
    else:
        print(f"🤖 Loading model ({DEVICE.upper()}) for query embedding…")
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)

    # ── Build hash matrices ───────────────────────────────────────
    pingala_mat = build_hash_matrix(pingala_fn, passages)
    plain_mat   = build_hash_matrix(plain_fn,   passages)
    print(f"\n✅ Ready — {len(passages)} passages indexed")
    print(f"   Type your question and press Enter.")
    print(f"   Type 'quit' or press Ctrl+C to exit.\n")

    # ── Query loop ────────────────────────────────────────────────
    while True:
        try:
            query = input("❓ Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query: continue
        if query.lower() in ("quit", "exit", "q"): break

        if DEVICE == "cuda": torch.cuda.empty_cache()
        t0     = time.perf_counter()
        q_emb  = model.encode([query], normalize_embeddings=True,
                               convert_to_numpy=True)[0]
        embed_ms = (time.perf_counter() - t0) * 1000

        # Pingala cascade
        t0 = time.perf_counter()
        p_ids, p_scores = cascade(query, q_emb, pingala_mat, pingala_fn, embs, top_k)
        pingala_ms = (time.perf_counter() - t0) * 1000

        # Brute force
        t0 = time.perf_counter()
        b_ids, b_scores = brute_force(q_emb, embs, top_k)
        bf_ms = (time.perf_counter() - t0) * 1000

        # ── Print results ─────────────────────────────────────────
        print(f"\n  Query embedded in {embed_ms:.1f}ms\n")

        print(f"  ── Pingala cascade ({pingala_ms:.1f}ms) " + "─" * 30)
        for rank, (idx, score) in enumerate(zip(p_ids, p_scores), 1):
            print(fmt_passage(passages[idx], score, rank))
            print()

        print(f"  ── Brute-force 768-dim ({bf_ms:.1f}ms) " + "─" * 27)
        for rank, (idx, score) in enumerate(zip(b_ids, b_scores), 1):
            print(fmt_passage(passages[idx], score, rank))
            print()

        # Flag if Pingala and brute-force agree on #1
        agree = p_ids[0] == b_ids[0] if len(p_ids) > 0 and len(b_ids) > 0 else False
        print(f"  Top result match: {'✅ agree' if agree else '⚠️  differ'} | "
              f"Pingala {pingala_ms:.1f}ms vs BF {bf_ms:.1f}ms "
              f"({bf_ms/max(pingala_ms,0.01):.1f}x speedup)\n")

if __name__ == "__main__":
    main()
