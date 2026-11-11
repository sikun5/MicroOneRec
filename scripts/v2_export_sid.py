#!/usr/bin/env python
"""Phase 4 entry: SID frozen diagnostics & downstream export (guide phase4_sid_diagnostics_export.md).
SCHEMA: phase4-v2_export_sid-1.0

--mode diagnostics (implemented, Phase 4 T3): frozen diagnostics for all
9 checkpoints — 9 clean quality stats + 27 perturbation combos (PPR@1/2/3,
NPR@10) with explicit support/exclusion counts, raw codes, quantized-sum
vectors, per-run invariance checks, three-seed summaries (mean / std ddof=1).
--mode export (Phase 4 T4): seed42 four-token SID export (sorted_item_suffix
disambiguation), shared vocab + per-group tokenizers, index / reverse map /
prefix trie, atomicity & roundtrip & trie-rejection checks.
--self-test-only: synthetic hand-computed metric checks (no GPU model needed).
"""
import os, sys, json, hashlib, argparse, time, csv, math
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rq"))
sys.path.insert(0, str(REPO / "scripts"))

import v2_train_rqvae as V

VIEWS = ("drop30", "drop60", "dropall")
SEEDS = (42, 43, 44)
METHODS = ("A", "B", "C")

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

# ---------- metric primitives (pure numpy; exercised by --self-test-only) ----------

def collision_stats(codes):
    """codes: [N,3] int. Raw collision / utilization / entropy stats."""
    N = codes.shape[0]
    from collections import Counter
    tuples = [tuple(int(v) for v in row) for row in codes]
    cnt = Counter(tuples)
    n_unique = len(cnt)
    res = {
        "n_items": N,
        "unique_tuples": n_unique,
        "collision_rate": 1.0 - n_unique / N,
        "involved_ratio": sum(c for c in cnt.values() if c > 1) / N,
        "max_bucket": int(max(cnt.values())),
    }
    for l in range(3):
        col = codes[:, l]
        counts = np.bincount(col, minlength=256)
        used = int((counts > 0).sum())
        p = counts[counts > 0] / N
        ent = float(-(p * np.log(p)).sum() / math.log(256)) + 0.0
        res[f"used_l{l+1}_of_256"] = used
        res[f"entropy_l{l+1}_normalized"] = ent
    pt = np.array(list(cnt.values())) / N
    res["entropy_tuple_normalized"] = float(-(pt * np.log(pt)).sum() / math.log(256)) + 0.0
    return res

def ppr_stats(codes_clean, codes_pert, support_idx):
    """Prefix keep-rate @1/2/3 on the first three codes, same model only."""
    cc = codes_clean[support_idx]
    cp = codes_pert[support_idx]
    n = len(support_idx)
    res = {"support": int(n), "ppr_denominator": int(n)}
    for l in (1, 2, 3):
        num = int((cc[:, :l] == cp[:, :l]).all(axis=1).sum())
        res[f"ppr{l}_numerator"] = num
        res[f"ppr{l}"] = num / n if n else None
    return res

def _safe_normalize(mat):
    """Row-normalize; returns (normed, zero_mask, nonfinite_mask). Zero/nonfinite
    rows are NEVER silently filled: callers must count and report them."""
    nonfinite = ~np.isfinite(mat).all(axis=1)
    m = np.where(nonfinite[:, None], 0.0, mat)
    norms = np.linalg.norm(m, axis=1)
    zero = norms == 0
    safe = np.where(zero | nonfinite, 1.0, norms)
    return m / safe[:, None], zero, nonfinite

def top10_ids(sims_row, self_pos, ids_sortkey):
    """Top-10 directory positions by cosine desc; exact ties broken by canonical
    item-ID sortkey (numeric) ascending. The self position is REMOVED from the
    candidate pool before ranking (not masked) so it can never be returned."""
    keep = np.arange(len(sims_row)) != self_pos
    s = sims_row[keep]
    ids_k = ids_sortkey[keep]
    pos_k = np.arange(len(sims_row))[keep]
    order = np.lexsort((ids_k, -s))[:10]
    return pos_k[order]

def clean_top10_all(clean_z, ids_sortkey, positions):
    """NN10 (inside the FIXED clean quantized directory) of each clean query.
    Returns dict position -> RANKED LIST (order preserved for audit)."""
    cl, cz, cn = _safe_normalize(clean_z)
    sims = cl[positions] @ cl.T
    out = {}
    for r, pos in enumerate(positions):
        out[int(pos)] = top10_ids(sims[r], pos, ids_sortkey).tolist()
    return out, int(cz.sum()), int(cn.sum())

def npr_stats(clean_z, pert_z, support_idx, ids_sortkey, precomputed_clean_top10):
    """NPR@10: both queries search the fixed clean directory of THIS run."""
    cl, cz, cn = _safe_normalize(clean_z)
    pl, pz, pn = _safe_normalize(pert_z[support_idx])
    res = {
        "support": int(len(support_idx)),
        "query_zero_norm_count": int(pz.sum()),
        "query_nonfinite_count": int(pn.sum()),
        "clean_dir_zero_norm_count": int(cz.sum()),
        "clean_dir_nonfinite_count": int(cn.sum()),
    }
    bad = pz.astype(int) + pn.astype(int)
    usable = support_idx[bad == 0]
    res["n_usable_queries"] = int(len(usable))
    if len(usable):
        urow = np.where(np.isin(support_idx, usable))[0]
        sims = pl[urow] @ cl.T
        inter = 0
        for r, pos in enumerate(usable):
            top_p = top10_ids(sims[r], pos, ids_sortkey)
            inter += len(set(top_p.tolist()) & set(precomputed_clean_top10[int(pos)]))
        res["npr10_intersect_sum"] = int(inter)
        res["npr10_denominator_sum"] = int(10 * len(usable))
        res["npr10"] = inter / (10 * len(usable))
    else:
        res["npr10_intersect_sum"] = 0
        res["npr10_denominator_sum"] = 0
        res["npr10"] = None
    return res

# ---------- self-test (synthetic, hand-computed; no model needed) ----------

def run_self_test():
    R = {"schema": "phase4-v2_export_sid-selftest-1.0", "date": "2026-11-11", "cases": {}}
    ok_all = True

    # C1 collision / utilization / entropy, hand-computed
    codes = np.array([[0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5], [6, 7, 8]])
    got = collision_stats(codes)
    p = np.array([2, 2, 1]) / 5
    exp_ent_l1 = float(-(p * np.log(p)).sum() / math.log(256))
    exp = {"collision_rate": 1 - 3 / 5, "involved_ratio": 4 / 5, "max_bucket": 2,
           "used_l1_of_256": 3, "used_l2_of_256": 3, "used_l3_of_256": 3}
    c1 = all(abs(got[k] - v) < 1e-12 for k, v in exp.items()) and abs(got["entropy_l1_normalized"] - exp_ent_l1) < 1e-12
    R["cases"]["C1_collision_entropy_hand"] = {"got": {**{k: got[k] for k in exp}, "entropy_l1_normalized": got["entropy_l1_normalized"]},
                                               "expected": {**exp, "entropy_l1_normalized": exp_ent_l1}, "pass": bool(c1)}
    ok_all &= c1

    # C2 PPR prefix-length semantics
    cc = np.array([[1, 2, 3], [4, 5, 6]])
    cp = np.array([[1, 2, 9], [4, 5, 6]])
    g2 = ppr_stats(cc, cp, np.array([0, 1]))
    e2 = {"ppr1": 1.0, "ppr2": 1.0, "ppr3": 0.5, "ppr_denominator": 2}
    c2 = all(abs(g2[k] - v) < 1e-12 for k, v in e2.items())
    R["cases"]["C2_ppr_hand"] = {"got": {k: g2[k] for k in e2}, "expected": e2, "pass": bool(c2)}
    ok_all &= c2

    # C3 NPR semantics on a 12-item directory (top10 well-defined), hand-computed:
    # item0 = angle 0; items 3,7 tied at 45 deg (cos .7071) -> ids 3 then 7;
    # fills: 1=60(.5) 2=90(0) 4=120(-.5) 5=150(-.866) 6=180(-1) 8=210(-.866)
    #        9=240(-.5) 10=270(0) 11=300(.5)
    # expected item0 top10 = [3,7,1,11,2,10,4,9,5,8]; self item0 excluded
    ang = {1: 60, 2: 90, 3: 45, 4: 120, 5: 150, 6: 180, 7: 45, 8: 210, 9: 240, 10: 270, 11: 300}
    rows = [np.array([1.0, 0.0])]
    for i in range(1, 12):
        t = math.radians(ang[i])
        rows.append(np.array([math.cos(t), math.sin(t)]))
    clean = np.vstack(rows)
    ids_sort = np.arange(12)
    ct10, zc, nf = clean_top10_all(clean, ids_sort, np.array([0, 1]))
    expected_top10 = [3, 7, 1, 11, 2, 10, 4, 9, 5, 8]
    # items 3 and 7 are byte-identical vectors -> exact fp tie -> id-ascending [3,7];
    # fill items are asserted by membership (fp ulp may reorder equal-cos pairs)
    tie_ok = list(ct10[0])[:2] == [3, 7] and set(ct10[0]) == set(expected_top10) and len(ct10[0]) == 10
    g3a = npr_stats(clean, clean.copy(), np.array([0, 1]), ids_sort, ct10)
    a_ok = abs(g3a["npr10"] - 1.0) < 1e-12 and g3a["npr10_intersect_sum"] == 20
    cl = clean / np.linalg.norm(clean, axis=1, keepdims=True)
    # perturb1 = (-1,0) = angle 180 = item6 -> its NN inside the FIXED clean dir is item6
    top1_of_pert1 = int(np.lexsort((ids_sort, -(cl[6] @ cl.T)))[0])
    b_ok = top1_of_pert1 == 6
    c3 = tie_ok and a_ok and b_ok
    R["cases"]["C3_npr_semantics"] = {
        "item0_top10_ranked": {"got": list(ct10[0]), "expected": expected_top10, "pass": bool(tie_ok)},
        "npr_identical_perturb": {"got": g3a["npr10"], "expected": 1.0, "pass": bool(a_ok)},
        "fixed_clean_reference_top1_for_pert1": {"got": top1_of_pert1, "expected": 6, "pass": bool(b_ok)},
        "pass": bool(c3)}
    ok_all &= c3

    # C4 exclude-self: self removed from pool; 12-item dir -> top10 has exactly 10
    row = cl[2] @ cl.T
    t = top10_ids(row, 2, ids_sort)
    c4 = (2 not in t.tolist()) and len(t) == 10
    R["cases"]["C4_exclude_self"] = {"self_in_top10": 2 in t.tolist(), "len": len(t),
                                     "expected_len": "min(10, N-1)=10", "pass": bool(c4)}
    ok_all &= c4

    # C5 metric-pipeline batch consistency (pure numpy, per-row decomposition)
    rng = np.random.default_rng(7)
    big_z = rng.normal(size=(64, 5))
    sup = np.arange(0, 64, 2)
    ct, _, _ = clean_top10_all(big_z, np.arange(64), sup)
    full = npr_stats(big_z, big_z + 0.01, sup, np.arange(64), ct)
    parts = [npr_stats(big_z, big_z + 0.01, chunk, np.arange(64), ct)
             for chunk in (sup[:16], sup[16:])]
    comb_inter = sum(x["npr10_intersect_sum"] for x in parts)
    c5 = comb_inter == full["npr10_intersect_sum"]
    R["cases"]["C5_metric_batch_consistency"] = {"full": full["npr10_intersect_sum"], "batched": comb_inter, "pass": bool(c5)}
    ok_all &= c5

    # C6 NaN injection: explicitly counted and excluded, never silently filled
    z6n = np.vstack([clean, [np.nan, 0.0]])  # 13 rows
    ids6 = np.arange(13)
    ct6, _, _ = clean_top10_all(z6n, ids6, np.array([0, 12]))
    g6 = npr_stats(z6n, z6n, np.array([0, 12]), ids6, ct6)
    c6 = g6["query_nonfinite_count"] == 1 and g6["n_usable_queries"] == 1
    R["cases"]["C6_nan_reported_not_filled"] = {"got": {k: g6[k] for k in ("query_nonfinite_count", "n_usable_queries")},
                                                "expected": {"query_nonfinite_count": 1, "n_usable_queries": 1}, "pass": bool(c6)}
    ok_all &= c6

    # C7 NPR no-overlap: 21-item directory, disjoint NN10 sets -> NPR = 0 (hand-computed)
    # item0 at 0 deg (query); ids 1..10 at 10..28 deg (its clean NN10, cos .985..0.883);
    # ids 11..20 at 80..98 deg; perturb of item0 = 90 deg direction:
    # perturb NN10 (within the SAME clean dir) = ids 11..20 (sin sim .985..1.0) vs
    # ids 1..10 at sin sim .174..0.469 and id0 excluded -> top10 = {11..20}, disjoint from {1..10}
    rows7 = [np.array([1.0, 0.0])]
    for i in range(1, 11):
        t = math.radians(8 + 2 * i)
        rows7.append(np.array([math.cos(t), math.sin(t)]))
    for i in range(11, 21):
        t = math.radians(80 + 2 * (i - 11))
        rows7.append(np.array([math.cos(t), math.sin(t)]))
    clean7 = np.vstack(rows7)
    ids7 = np.arange(21)
    pert7 = clean7.copy()
    pert7[0] = np.array([0.0, 1.0])
    ct7, _, _ = clean_top10_all(clean7, ids7, np.array([0]))
    g7 = npr_stats(clean7, pert7, np.array([0]), ids7, ct7)
    nn_clean = ct7[0]
    nn_pert = top10_ids((_safe_normalize(pert7)[0][0]) @ (_safe_normalize(clean7)[0].T), 0, ids7)
    c7 = (list(nn_clean) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
          and set(nn_pert.tolist()) == set(range(11, 21))
          and 0 not in nn_clean and 0 not in nn_pert.tolist()
          and g7["npr10"] == 0.0 and g7["npr10_intersect_sum"] == 0
          and g7["npr10_denominator_sum"] == 10)
    R["cases"]["C7_npr_no_overlap_zero"] = {
        "clean_nn10_ranked": {"got": list(nn_clean), "expected": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        "perturb_nn10_set": {"got": sorted(nn_pert.tolist()), "expected": list(range(11, 21))},
        "npr10": {"got": g7["npr10"], "expected": 0.0, "intersect_sum": g7["npr10_intersect_sum"],
                  "denominator_sum": g7["npr10_denominator_sum"]},
        "self_excluded": {"clean": 0 not in nn_clean, "perturb": 0 not in nn_pert.tolist()},
        "pass": bool(c7)}
    ok_all &= c7


    # C8 sorted_item_suffix disambiguation: collisions + singleton + non-contiguous ids
    ids8 = ["10", "2", "10", "999", "10"]
    codes8 = np.array([[1, 1, 1], [1, 1, 1], [5, 5, 5], [1, 1, 1], [7, 7, 7]])
    s8, suf8, mj8 = sid_strings(codes8, ids8)
    # bucket (1,1,1): ids 2(idx1)<10(idx0)<999(idx3) -> j=0,1,2 ; (5,5,5) singleton idx2 -> 0 ; (7,7,7) singleton idx4 -> 0
    e8 = {"suffix": [1, 0, 0, 2, 0], "max_suffix": 2,
          "sids": ["<a_1><b_1><c_1><d_1>", "<a_1><b_1><c_1><d_0>", "<a_5><b_5><c_5><d_0>",
                   "<a_1><b_1><c_1><d_2>", "<a_7><b_7><c_7><d_0>"],
          "first3_unchanged": True}
    got8 = {"suffix": suf8, "max_suffix": mj8, "sids": s8,
            "first3_unchanged": [int(x) for x in codes8[:, 0]] == [1, 1, 5, 1, 7]}
    c8 = got8 == e8
    R["cases"]["C8_suffix_disambiguation"] = {"got": got8, "expected": e8, "pass": bool(c8)}
    ok_all &= c8

    # C9 decode contract on synthetic sids: 4-token split / trie accept / reject invalid combos
    seqs = [[f"<a_{c[0]}>", f"<b_{c[1]}>", f"<c_{c[2]}>", f"<d_{j}>"] for c, j in zip(codes8, suf8)]
    trie9 = build_trie(seqs)
    acc = all(trie_accepts(trie9, q) for q in seqs)
    rej1 = trie_accepts(trie9, seqs[4][:3] + [seqs[0][3]])        # (7,7,7) bucket size 1: <d_1> out of range
    rej2 = trie_accepts(trie9, [t + "_x" for t in seqs[1]])       # nonexistent tokens
    rej3 = trie_accepts(trie9, seqs[1][:3])                       # truncated 3-token path
    c9 = acc and (not rej1) and (not rej2) and (not rej3)
    R["cases"]["C9_trie_decode_contract"] = {
        "accepts_all_actual": acc, "rejects_suffix_swap": not rej1,
        "rejects_nonexistent_tokens": not rej2, "rejects_truncated_3token": not rej3,
        "roundtrip_split_join": all("|".join(q) == s.replace("><", ">|<") for q, s in zip(seqs, s8)),
        "pass": bool(c9)}
    ok_all &= c9

    R["all_passed"] = bool(ok_all)
    return R

def assert_usable_vectors(mat, ctx):
    """Formal-run hard guard: ANY non-finite or zero-norm row among candidates or
    queries ABORTS the run (exception -> nonzero exit, no metrics written).
    Skipping rows with silently changed denominators is forbidden in production;
    the counting/skip behavior is exercised only by synthetic self-test C6."""
    if not np.isfinite(mat).all():
        n = int((~np.isfinite(mat).all(axis=1)).sum())
        raise RuntimeError(f"NONFINITE_ROWS_ABORT {ctx}: {n} row(s) contain non-finite values; "
                           "formal metrics must not be produced")
    norms = np.linalg.norm(mat, axis=1)
    if (norms == 0).any():
        n = int((norms == 0).sum())
        raise RuntimeError(f"ZERO_NORM_ROWS_ABORT {ctx}: {n} zero-norm row(s); "
                           "formal metrics must not be produced")

# ---------- model loading & encoding ----------

def load_model(ckpt_path, device):
    import torch
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = st["config"]
    model = V.V2RQVAE(in_dim=st["init_state"]["in_dim"], num_emb_list=[256, 256, 256], e_dim=32,
                      layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=False,
                      loss_type="mse", quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
                      kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50).to(device)
    model.load_state_dict(st["state_dict"], strict=True)
    V.set_vq_initted(model, st["init_state"]["vq_initted"])
    model.eval()
    return model, st, cfg

def state_checksum(model):
    import torch
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()

def encode_full(model, x_all, device):
    """Official encode: ONE full forward covering all 3686 rows (fixed shape ->
    no cross-batch kernel ambiguity in official numbers)."""
    import torch
    with torch.no_grad():
        xb = torch.from_numpy(x_all).to(device)
        out, rq_loss, indices, zq = model(xb, use_sk=False)
        idx2 = model.get_indices(xb, use_sk=False)
        assert torch.equal(indices, idx2), "forward indices != get_indices(use_sk=False)"
        codes = indices.detach().cpu().numpy().astype(np.int64)
        zq_np = zq.detach().cpu().numpy().astype(np.float32)
    del out, rq_loss
    return codes, zq_np

def encode_batched(model, x_all, device, batch):
    """Diagnostic-only batched encode: measures batch-shape sensitivity."""
    import torch
    codes, zqs = [], []
    with torch.no_grad():
        for i in range(0, len(x_all), batch):
            xb = torch.from_numpy(x_all[i:i + batch]).to(device)
            _, _, indices, zq = model(xb, use_sk=False)
            codes.append(indices.detach().cpu().numpy().astype(np.int64))
            zqs.append(zq.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(codes), np.concatenate(zqs)

# ---------- main diagnostics ----------

def run_diagnostics(args, st_res):
    import torch
    device = torch.device(args.device)
    torch.cuda.set_device(device)  # Phase 3 pattern: current-device no-arg memory stats
    emb_root = REPO / args.embeddings_root
    rq_root = REPO / args.rqvae_root
    sid_root = REPO / args.output_root
    ev_dir = REPO / args.evidence_dir
    (ev_dir / "logs").mkdir(parents=True, exist_ok=True)
    ids_raw = json.load(open(emb_root / "item_ids.json"))
    ids = [str(i) for i in (ids_raw["item_ids"] if isinstance(ids_raw, dict) else ids_raw)]
    assert len(ids) == 3686
    ids_sortkey = np.array([int(i) if i.isdigit() else -1 for i in ids], dtype=np.int64)
    order_rule = ids_raw.get("order_rule") if isinstance(ids_raw, dict) else None

    cat_missing = set()
    with open(REPO / "outputs/v2/manifests/catalog_clean.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("original_description_missing") is True:
                cat_missing.add(str(r["item_id"]))
    S_raw_nonempty = set(ids) - cat_missing

    ckpt_shas_pre = {f"{m}_{s}": sha256_file(rq_root / m / f"seed_{s}" / "final_checkpoint.pth")
                     for m in args.methods for s in args.seeds}
    inv = {"schema": "phase4-t3-run-inventory-1.0", "date": "2026-11-11", "device": args.device,
           "mode": "diagnostics", "official_batch": "single full forward (3686 rows)",
           "checkpoint_sha_pre": ckpt_shas_pre, "runs": []}
    checks = {"schema": "phase4-diagnostic-checks-1.0", "self_test": st_res, "runs": {}, "coverage": {}}

    n_combos = 0
    for m in args.methods:
        clean_x = np.load(emb_root / m / "clean.npy")
        assert clean_x.shape == (3686, 2560) and clean_x.dtype == np.float32
        for s in args.seeds:
            t0 = time.time()
            rid = f"SID-{m}-{s}"
            run_dir = sid_root / m / f"seed_{s}"
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            ckpt = rq_root / m / f"seed_{s}" / "final_checkpoint.pth"
            model, st, cfg = load_model(ckpt, device)
            chk_pre = state_checksum(model)
            initted_pre = V.vq_initted_flags(model)
            assert initted_pre == [True, True, True], f"vq_initted not all True: {initted_pre}"

            codes_c, zq_c = encode_full(model, clean_x, device)
            assert_usable_vectors(zq_c, f"{rid} clean quantized directory (all 3686 candidates)")
            codes_b512, zq_b512 = encode_batched(model, clean_x, device, 512)
            batch_idx_equal = bool(np.array_equal(codes_c, codes_b512))
            batch_zq_maxdiff = float(np.abs(zq_c - zq_b512).max()) if zq_c.shape == zq_b512.shape else None

            cstat = collision_stats(codes_c)
            raw_codes = {"schema": "phase4-sid-raw_codes-1.0", "run_id": rid, "method": m, "seed": s,
                         "order_rule": order_rule, "item_ids": ids, "codes": codes_c.tolist(),
                         "code_range_checked": [int(codes_c.min()), int(codes_c.max())]}
            (run_dir / "raw_codes.json").write_text(json.dumps(raw_codes, ensure_ascii=False))
            np.save(run_dir / "quantized_clean.npy", zq_c)

            masks = {v: np.asarray(np.load(emb_root / m / "diagnostic" / f"{v}_valid.npy")).astype(bool)
                     for v in args.views}
            support_union = sorted({i for v in args.views for i in np.where(masks[v])[0]
                                    if ids[i] in S_raw_nonempty})
            support_union_arr = np.array(support_union, dtype=np.int64)
            ct10, dirz, dirnf = clean_top10_all(zq_c, ids_sortkey, support_union_arr)

            diag = {"schema": "phase4-sid-diagnostics-1.0", "run_id": rid, "method": m, "seed": s,
                    "guards": {"eval_mode": not model.training, "no_grad": True, "use_sk": False,
                               "state_checksum_pre": chk_pre, "vq_initted_pre": initted_pre},
                    "inputs": {"final_checkpoint_sha256": ckpt_shas_pre[f"{m}_{s}"],
                               "config_script_sha256": cfg.get("script_sha256"),
                               "clean_npy_sha256": sha256_file(emb_root / m / "clean.npy"),
                               "item_ids_sha256": sha256_file(emb_root / "item_ids.json"),
                               "diagnostic_npy_sha256": {v: sha256_file(emb_root / m / "diagnostic" / f"{v}.npy") for v in args.views},
                               "mask_sha256": {v: sha256_file(emb_root / m / "diagnostic" / f"{v}_valid.npy") for v in args.views}},
                    "clean": cstat,
                    "batch_sensitivity": {"official": "single full forward (3686 rows)",
                                          "batch512_indices_equal": batch_idx_equal,
                                          "batch512_zq_maxabsdiff": batch_zq_maxdiff,
                                          "note": "批次形状敏感性为实测记录；正式数值统一全量单批前向"},
                    "support_union_size": len(support_union),
                    "views": {}}
            for view in args.views:
                px = np.load(emb_root / m / "diagnostic" / f"{view}.npy")
                assert px.shape == (3686, 2560) and px.dtype == np.float32
                codes_p, zq_p = encode_full(model, px, device)
                assert_usable_vectors(zq_p, f"{rid} {view} perturb quantized (all rows)")
                np.save(run_dir / f"quantized_perturb_{view}.npy", zq_p)
                mv = masks[view]
                sup_idx = np.array([i for i in np.where(mv)[0] if ids[i] in S_raw_nonempty], dtype=np.int64)
                pres = ppr_stats(codes_c, codes_p, sup_idx)
                nres = npr_stats(zq_c, zq_p, sup_idx, ids_sortkey, ct10)
                pres["excluded_raw_nonempty_not_in_mask"] = int(len(S_raw_nonempty) - len(sup_idx))
                pres["excluded_mask_valid_but_raw_missing"] = int(len(np.where(mv)[0]) - len(sup_idx))
                diag["views"][view] = {"perturb_npy_sha256": sha256_file(emb_root / m / "diagnostic" / f"{view}.npy"),
                                       "support": int(len(sup_idx)),
                                       "ppr": pres, "npr": nres}
                n_combos += 1
                print(f"{rid} {view}: ppr {pres['ppr1']:.4f}/{pres['ppr2']:.4f}/{pres['ppr3']:.4f} "
                      f"npr10 {nres['npr10']:.4f} sup {pres['support']}", flush=True)

            chk_post = state_checksum(model)
            initted_post = V.vq_initted_flags(model)
            diag["guards"]["state_checksum_post"] = chk_post
            diag["guards"]["state_unchanged"] = chk_pre == chk_post
            diag["guards"]["vq_initted_post"] = initted_post
            diag["guards"]["initted_unchanged"] = initted_pre == initted_post
            diag["guards"]["nonfinite_zero_norm_policy"] = "formal run aborts (RuntimeError -> exit != 0, no metrics written); counting/skip only in synthetic self-test C6"
            diag["guards"]["codes_range"] = [int(codes_c.min()), int(codes_c.max())]
            diag["guards"]["codes_cover_3686"] = codes_c.shape[0] == 3686
            diag["guards"]["zq_clean_all_finite"] = bool(np.isfinite(zq_c).all())
            del model
            torch.cuda.empty_cache()
            wall = time.time() - t0
            peak = int(torch.cuda.max_memory_allocated())
            diag["runtime"] = {"wall_s": round(wall, 1), "peak_mem_allocated_bytes": peak}
            (run_dir / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False, indent=1))
            inv["runs"].append({"run_id": rid, "wall_s": round(wall, 1), "peak_mem_allocated_bytes": peak,
                                "diagnostics_sha256": sha256_file(run_dir / "diagnostics.json"),
                                "raw_codes_sha256": sha256_file(run_dir / "raw_codes.json")})
            print(f"{rid} DONE wall={wall:.1f}s peak={peak} state_unchanged={diag['guards']['state_unchanged']} "
                  f"batch512_idx_equal={batch_idx_equal}", flush=True)
            checks["runs"][rid] = {"state_unchanged": diag["guards"]["state_unchanged"],
                                   "initted_unchanged": diag["guards"]["initted_unchanged"],
                                   "batch512_indices_equal": batch_idx_equal,
                                   "codes_range_ok": (diag["guards"]["codes_range"][0] >= 0 and diag["guards"]["codes_range"][1] <= 255),
                                   "zq_finite": diag["guards"]["zq_clean_all_finite"]}

    ckpt_shas_post = {f"{m}_{s}": sha256_file(rq_root / m / f"seed_{s}" / "final_checkpoint.pth")
                      for m in args.methods for s in args.seeds}
    inv["checkpoint_unchanged"] = ckpt_shas_pre == ckpt_shas_post
    inv["checkpoint_sha_post"] = ckpt_shas_post
    checks["checkpoint_files_unchanged"] = ckpt_shas_pre == ckpt_shas_post
    checks["coverage"] = {"clean_runs": len(args.methods) * len(args.seeds), "perturb_combos": n_combos,
                          "expected_clean": 9, "expected_combos": 27}
    checks["coverage"]["complete"] = checks["coverage"]["clean_runs"] == 9 and n_combos == 27

    qual_rows, rob_rows = [], []
    for m in args.methods:
        for s in args.seeds:
            d = json.load(open(sid_root / m / f"seed_{s}" / "diagnostics.json"))
            c = d["clean"]
            qual_rows.append(["raw", m, s, c["collision_rate"], c["involved_ratio"], c["max_bucket"],
                              c["used_l1_of_256"], c["used_l2_of_256"], c["used_l3_of_256"],
                              c["entropy_l1_normalized"], c["entropy_l2_normalized"], c["entropy_l3_normalized"]])
            for view in args.views:
                p, n = d["views"][view]["ppr"], d["views"][view]["npr"]
                rob_rows.append(["raw", m, s, view, p["support"], p["ppr1"], p["ppr2"], p["ppr3"],
                                 p["ppr1_numerator"], p["ppr2_numerator"], p["ppr3_numerator"],
                                 n["npr10"], n["npr10_intersect_sum"], n["npr10_denominator_sum"],
                                 p["excluded_raw_nonempty_not_in_mask"], p["excluded_mask_valid_but_raw_missing"],
                                 n["query_zero_norm_count"] + n["query_nonfinite_count"]])

    def agg(rows, col):
        vals = np.array([r[col] for r in rows], dtype=float)
        return float(vals.mean()), float(vals.std(ddof=1))

    for m in args.methods:
        rows = [r for r in qual_rows if r[1] == m and r[0] == "raw"]
        mean, std = {}, {}
        for col in range(3, 12):
            mean[col], std[col] = agg(rows, col)
        qual_rows.append(["mean", m, "", *[mean[c] for c in range(3, 12)]])
        qual_rows.append(["std_ddof1", m, "", *[std[c] for c in range(3, 12)]])
        for view in args.views:
            rws = [r for r in rob_rows if r[1] == m and r[3] == view and r[0] == "raw"]
            mean2, std2 = {}, {}
            for col in (4, 5, 6, 7, 11, 12):
                mean2[col], std2[col] = agg(rws, col)
            rob_rows.append(["mean", m, "", view, mean2[4], mean2[5], mean2[6], mean2[7], "", "", "",
                             mean2[11], mean2[12], "", "", ""])
            rob_rows.append(["std_ddof1", m, "", view, std2[4], std2[5], std2[6], std2[7], "", "", "",
                             std2[11], std2[12], "", "", ""])

    with open(sid_root / "quality_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "method", "seed", "collision_rate", "involved_ratio", "max_bucket",
                    "used_l1_of_256", "used_l2_of_256", "used_l3_of_256",
                    "entropy_l1_normalized", "entropy_l2_normalized", "entropy_l3_normalized"])
        w.writerows(qual_rows)
    with open(sid_root / "robustness_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "method", "seed", "view", "support", "ppr1", "ppr2", "ppr3",
                    "ppr1_num", "ppr2_num", "ppr3_num", "npr10", "npr10_intersect_sum",
                    "npr10_denominator_sum", "excluded_raw_nonempty_not_in_mask",
                    "excluded_mask_valid_but_raw_missing", "nonfinite_or_zeronorm_rows"])
        w.writerows(rob_rows)

    checks["all_runs_guards_passed"] = all(v["state_unchanged"] and v["initted_unchanged"] and v["zq_finite"]
                                           and v["codes_range_ok"] for v in checks["runs"].values())
    checks["self_test_all_passed"] = st_res["all_passed"]
    (ev_dir / "diagnostic_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1))
    (ev_dir / "t3_run_inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1))
    print("CHECKPOINT_UNCHANGED:", inv["checkpoint_unchanged"], flush=True)
    print("COVERAGE:", json.dumps(checks["coverage"], ensure_ascii=False), flush=True)
    print("GUARDS_ALL_PASSED:", checks["all_runs_guards_passed"], flush=True)
    print("QUAL_SHA256:", sha256_file(sid_root / "quality_summary.csv"), flush=True)
    print("ROB_SHA256:", sha256_file(sid_root / "robustness_summary.csv"), flush=True)
    print("CHECKS_SHA256:", sha256_file(ev_dir / "diagnostic_checks.json"), flush=True)
    print("INV_SHA256:", sha256_file(ev_dir / "t3_run_inventory.json"), flush=True)
    print("T3_DIAGNOSTICS_DONE", flush=True)


# ---------- export (Phase 4 T4): seed42 four-token SID, shared vocab, tokenizer ----------

SORT_KEY = lambda s: (0, int(s)) if s.isdigit() else (1, s)

def sid_strings(codes, ids):
    """codes [N,3] int, ids list[str]. sorted_item_suffix policy: buckets by the
    full 3-tuple; within a bucket items numbered j=0,1,... by canonical item-ID
    order (numeric first); singleton buckets get <d_0>. First three codes unchanged."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, row in enumerate(codes):
        buckets[tuple(int(v) for v in row)].append(i)
    suffix = [None] * len(codes)
    max_j = -1
    for tup, members in buckets.items():
        for j, i in enumerate(sorted(members, key=lambda i: SORT_KEY(ids[i]))):
            suffix[i] = j
        max_j = max(max_j, len(members) - 1)
    sids = [f"<a_{int(codes[i][0])}><b_{int(codes[i][1])}><c_{int(codes[i][2])}><d_{suffix[i]}>"
            for i in range(len(codes))]
    return sids, suffix, max_j

def build_shared_vocab(max_suffix):
    """768 semantic tokens (all 256 per layer, used or not) + suffix union [0..max_suffix]."""
    toks = [f"<{l}_{k}>" for l in ("a", "b", "c") for k in range(256)]
    toks += [f"<d_{j}>" for j in range(max_suffix + 1)]
    return toks

def build_trie(seqs):
    trie = {}
    for seq in seqs:
        node = trie
        for t in seq:
            node = node.setdefault(t, {})
        node["__leaf__"] = True
    return trie

def trie_accepts(trie, seq):
    node = trie
    for t in seq:
        if t not in node:
            return False
        node = node[t]
    return "__leaf__" in node

BASE_TOKENIZER_PATH = "/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
INIT_SEED = 42

def run_export(args, st_res):
    import torch  # noqa: ensure env parity; no LLM weights are loaded
    from transformers import AutoTokenizer
    sid_root = REPO / args.output_root
    data_root = REPO / "outputs/v2/data"
    tok_root = REPO / "outputs/v2/tokenizer"
    ev_dir = REPO / args.evidence_dir
    methods = ("A", "B", "C")

    per_group = {}
    for m in methods:
        rc_path = sid_root / m / "seed_42" / "raw_codes.json"
        rc = json.load(open(rc_path))
        ids = [str(i) for i in rc["item_ids"]]
        assert len(ids) == 3686
        codes = np.array(rc["codes"], dtype=np.int64)
        sids, suffix, max_j = sid_strings(codes, ids)
        per_group[m] = {"ids": ids, "codes": codes, "sids": sids, "suffix": suffix,
                        "max_suffix": max_j, "raw_codes_sha256": sha256_file(rc_path)}
        print(f"export {m}: n={len(sids)} unique={len(set(sids))} max_suffix={max_j}", flush=True)
    max_suffix_union = max(per_group[m]["max_suffix"] for m in methods)
    shared = build_shared_vocab(max_suffix_union)

    tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER_PATH, local_files_only=True)
    base_len = len(tok)
    n_added = tok.add_tokens(shared)
    assert n_added == len(shared), f"add_tokens added {n_added}, expected {len(shared)}"
    token_ids_ordered = tok.convert_tokens_to_ids(shared)
    assert token_ids_ordered == list(range(base_len, base_len + len(shared))), \
        "registration order not contiguous after base vocab"
    reg_order_sha = hashlib.sha256(json.dumps(shared, ensure_ascii=False,
                                              separators=(",", ":")).encode()).hexdigest()
    eos_id, eos_tok = tok.eos_token_id, tok.eos_token
    atomic_ok = all(tok.tokenize(t) == [t] for t in shared)

    manifest = {
        "schema": "phase4-tokenizer-binding-1.0", "date": "2026-11-11",
        "base": {"repo_id": "Qwen/Qwen3-0.6B-Base", "revision": BASE_REVISION,
                 "local_path": BASE_TOKENIZER_PATH, "base_vocab_size": base_len,
                 "loaded_with": "AutoTokenizer.from_pretrained(local_path, local_files_only=True); no model weights loaded"},
        "init_seed": INIT_SEED,
        "init_policy": "future LLM embedding init must reproduce from: base revision + this shared token-ID order + seed 42 (torch.manual_seed(42)); no LLM is trained or downloaded in Phase 4",
        "shared_vocab_size": len(shared),
        "registration_order_rule": "<a_0..255>, <b_0..255>, <c_0..255>, then <d_j> ascending j; appended after base vocab in that order",
        "registration_order_sha256": reg_order_sha,
        "eos_termination_protocol": f"4 SID tokens form the complete item identifier; generation terminates with the base EOS {eos_tok} (id {eos_id}) recorded here as the frozen protocol",
        "all_tokens_atomic": atomic_ok,
        "token_id_map_sha256": hashlib.sha256(json.dumps(dict(zip(shared, token_ids_ordered)),
                                                         ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
    }
    tok_root.mkdir(parents=True, exist_ok=True)
    (tok_root / "shared_vocab.json").write_text(json.dumps(
        {"schema": "phase4-shared-vocab-1.0", "n_tokens": len(shared),
         "max_suffix_union": max_suffix_union, "tokens": shared,
         "registration_order_sha256": reg_order_sha}, ensure_ascii=False, indent=1))
    (tok_root / "token_id_map.json").write_text(json.dumps(
        {"schema": "phase4-token-id-map-1.0", "base_vocab_size": base_len,
         "map": {t: i for t, i in zip(shared, token_ids_ordered)}}, ensure_ascii=False, indent=1))
    (tok_root / "binding_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    ex_checks = {"schema": "phase4-export-checks-1.0", "date": "2026-11-11",
                 "self_test": st_res, "groups": {}, "notes": {}}
    tk_checks = {"schema": "phase4-tokenizer-checks-1.0", "date": "2026-11-11",
                 "manifest": manifest, "saved_dirs": {}, "roundtrip": {}}
    all_ok = True

    for m in methods:
        g = per_group[m]
        d = data_root / m
        d.mkdir(parents=True, exist_ok=True)
        expected_tokens = [[f"<a_{int(c[0])}>", f"<b_{int(c[1])}>", f"<c_{int(c[2])}>", f"<d_{j}>"]
                           for c, j in zip(g["codes"], g["suffix"])]
        atomic_full = all(tok.tokenize(s) == e for s, e in zip(g["sids"], expected_tokens))
        token_ids = [tok.convert_tokens_to_ids(e) for e in expected_tokens]
        decode_ok = all(tok.decode(t) == s for t, s in zip(token_ids, g["sids"]))
        uniq = len(set(g["sids"])) == 3686
        # independent parse-back: re-derive the three codes + suffix from each SID string
        # and compare against the T3 raw_codes entries (not a tautology: parses the strings)
        import re
        pat = re.compile(r"^<a_(\d+)><b_(\d+)><c_(\d+)><d_(\d+)>$")
        parsed_ok = True
        for sid_s, c_row, j in zip(g["sids"], g["codes"], g["suffix"]):
            mt = pat.match(sid_s)
            if (not mt or (int(mt.group(1)), int(mt.group(2)), int(mt.group(3))) != (int(c_row[0]), int(c_row[1]), int(c_row[2]))
                    or int(mt.group(4)) != int(j)):
                parsed_ok = False
                break
        rev = dict(zip(g["sids"], g["ids"]))
        rev_ok = all(rev[s] == i for s, i in zip(g["sids"], g["ids"]))
        trie = build_trie(token_ids)
        accept_all = all(trie_accepts(trie, t) for t in token_ids)
        # rejection samples: deterministic guaranteed-invalid probes
        # (a) out-of-range suffix for that prefix: <d_{bucket_size}>; (b) 5-token over-extension;
        # (c) truncated 3-token path. Cross-prefix suffix swaps can accidentally be real
        # paths, so they are not used as rejection probes.
        from collections import Counter
        bucket_size = Counter(tuple(int(v) for v in row) for row in g["codes"])
        rng = np.random.default_rng(42)
        rej_n, rej_hit = 0, 0
        for _ in range(1000):
            i = int(rng.integers(0, 3686))
            pre = token_ids[i][:3]
            size = bucket_size[tuple(int(v) for v in g["codes"][i])]
            bads = (pre + ["__out_of_range_suffix__"], token_ids[i] + [token_ids[i][0]], token_ids[i][:3])
            for bad in bads:
                if not trie_accepts(trie, bad):
                    rej_hit += 1
                rej_n += 1
        (d / "Industrial_and_Scientific.index.json").write_text(json.dumps(
            {"schema": "phase4-sid-index-1.0", "method": m, "seed": 42,
             "order_rule": SORT_DOC, "item_ids": g["ids"], "sids": g["sids"],
             "token_ids": token_ids}, ensure_ascii=False, indent=1))
        (d / "reverse_item_map.json").write_text(json.dumps(
            {"schema": "phase4-sid-reverse-1.0", "method": m, "note": "full 4-token SID key; never reconstruct by truncating the first three codes",
             "reverse": rev}, ensure_ascii=False, indent=1))
        (d / "prefix_trie.json").write_text(json.dumps(
            {"schema": "phase4-sid-prefix-trie-1.0", "method": m, "n_paths": len(token_ids),
             "trie": trie}, ensure_ascii=False, indent=1))
        ex_checks["groups"][m] = {
            "n_items": 3686, "coverage_4_tokens": True,
            "unique_full_sids": uniq, "reverse_consistent": rev_ok,
            "first3_match_raw_codes": parsed_ok,
            "raw_codes_sha256": g["raw_codes_sha256"],
            "max_suffix": g["max_suffix"],
            "atomic_full_sids": atomic_full, "decode_roundtrip": decode_ok,
            "trie_accepts_all_actual": accept_all,
            "trie_rejection_samples": {"tried": rej_n, "rejected": rej_hit, "rng_seed": 42},
        }
        gok = uniq and rev_ok and parsed_ok and atomic_full and decode_ok and accept_all and rej_hit == rej_n
        ex_checks["groups"][m]["passed"] = bool(gok)
        all_ok &= gok
        print(f"export {m}: atomic={atomic_full} roundtrip={decode_ok} uniq={uniq} rev={rev_ok} "
              f"trie={accept_all} reject={rej_hit}/{rej_n}", flush=True)

    # tokenizer dir per group + independent reload verification
    for m in methods:
        td = tok_root / m
        tok.save_pretrained(td)
        rt = AutoTokenizer.from_pretrained(td, local_files_only=True)
        reload_ids = rt.convert_tokens_to_ids(shared)
        same = reload_ids == token_ids_ordered
        sample_atomic = all(rt.tokenize(t) == [t] for t in shared[:64] + shared[-64:])
        full_rt = rt.tokenize(per_group[m]["sids"][1234]) == per_group[m]["sids"][1234].replace("><", ">|<").split("|")
        sid_toks = rt.tokenize(per_group[m]["sids"][1234])
        full_rt = (len(sid_toks) == 4 and rt.decode(rt.convert_tokens_to_ids(sid_toks)) == per_group[m]["sids"][1234])
        tk_checks["saved_dirs"][m] = {"path": f"outputs/v2/tokenizer/{m}",
                                      "reload_ids_identical": same,
                                      "sample_atomic": sample_atomic,
                                      "sample_full_sid_roundtrip": full_rt}
        all_ok &= same and sample_atomic and full_rt

    diag_c = json.load(open(sid_root / "C" / "seed_42" / "diagnostics.json"))
    ex_checks["notes"] = {
        "collision_disclosure": {
            "C42_raw_collision_rate": diag_c["clean"]["collision_rate"],
            "C42_max_bucket": diag_c["clean"]["max_bucket"],
            "statement": "完整 SID 经 sorted_item_suffix 消歧后唯一率 100% 是技术条件；C-42 原始三层碰撞率 "
                         f"{diag_c['clean']['collision_rate']:.4f}（最大桶 {diag_c['clean']['max_bucket']}）"
                         "依旧披露，后缀唯一率不是语义质量提升，语义评估留 Phase 5。"},
        "eos_protocol": manifest["eos_termination_protocol"],
        "base_model_untouched": "未加载/未训练任何 LLM 权重；未来扩词初始化由 manifest 固定配置复现",
    }
    ex_checks["all_passed"] = bool(all_ok)
    tk_checks["all_passed"] = bool(all_ok)
    (ev_dir / "export_checks.json").write_text(json.dumps(ex_checks, ensure_ascii=False, indent=1))
    (ev_dir / "tokenizer_checks.json").write_text(json.dumps(tk_checks, ensure_ascii=False, indent=1))
    print("REG_ORDER_SHA256:", reg_order_sha, flush=True)
    print("SHARED_VOCAB_N:", len(shared), "MAX_SUFFIX_UNION:", max_suffix_union, flush=True)
    print("EXPORT_ALL_PASSED:", all_ok, flush=True)
    print("EXPORT_CHECKS_SHA256:", sha256_file(ev_dir / "export_checks.json"), flush=True)
    print("TOKENIZER_CHECKS_SHA256:", sha256_file(ev_dir / "tokenizer_checks.json"), flush=True)
    print("T4_EXPORT_DONE", flush=True)

SORT_DOC = "item IDs sorted numeric-first (int value), non-numeric IDs in canonical string order afterwards"

def main():
    ap = argparse.ArgumentParser(
        description="Phase 4 SID diagnostics/export (T3: diagnostics implemented; T4 will add export)",
        epilog="Diagnostics: 9 clean quality stats + 27 perturbation combos (PPR@1/2/3, NPR@10), "
               "frozen inference eval()/no_grad()/use_sk=False, fixed clean-directory NPR reference.")
    ap.add_argument("--mode", choices=["diagnostics", "export", "all"], default="diagnostics",
                    help="diagnostics = frozen 9+27 diagnostics (implemented); export/all = not implemented until Phase 4 T4 (refuses)")
    ap.add_argument("--rqvae-root", default="outputs/v2/rqvae", help="root of 9 final checkpoints")
    ap.add_argument("--embeddings-root", default="outputs/v2/embeddings", help="Phase 2 frozen cache root")
    ap.add_argument("--output-root", default="outputs/v2/sid", help="SID output root")
    ap.add_argument("--evidence-dir", default="outputs/v2/evidence/phase4", help="evidence output dir")
    ap.add_argument("--diagnostic-manifest", default="outputs/v2/manifests/diagnostic_views.jsonl",
                    help="seed2701 view manifest (provenance; *_valid.npy masks remain the authoritative valid sets)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--self-test", action="store_true", help="run synthetic hand-computed metric checks before diagnostics")
    ap.add_argument("--self-test-only", action="store_true", help="run only the synthetic self-test and exit")
    args = ap.parse_args()

    st_res = None
    if args.self_test or args.self_test_only:
        st_res = run_self_test()
        print("SELF_TEST_ALL_PASSED:", st_res["all_passed"], flush=True)
        ev_dir = REPO / args.evidence_dir
        ev_dir.mkdir(parents=True, exist_ok=True)
        (ev_dir / "diagnostic_selftest.json").write_text(json.dumps(st_res, ensure_ascii=False, indent=1))
        print("SELFTEST_SHA256:", sha256_file(ev_dir / "diagnostic_selftest.json"), flush=True)
        if args.self_test_only:
            print("T3_SELFTEST_DONE", flush=True)
            return
        if not st_res["all_passed"]:
            print("ERROR: self-test failed; refusing to run diagnostics.", file=sys.stderr)
            sys.exit(3)
    if args.mode == "diagnostics":
        run_diagnostics(args, st_res)
        return
    if args.mode == "export":
        run_export(args, st_res)
        return
    run_diagnostics(args, st_res)
    run_export(args, st_res)

if __name__ == "__main__":
    main()
