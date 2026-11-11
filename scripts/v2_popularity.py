#!/usr/bin/env python
"""Phase 5 T4: B0 Popularity baseline - build, freeze, and validation evaluation.
SCHEMA: phase5-v2_popularity-1.0

PROTOCOL (boss 2026-11-11):
- Frequencies from TRAIN targets ONLY (36,259); the builder never touches
  valid/test (enforced by code path: build() takes only the train path).
- Complete 3,686-item catalog ranking: frequency DESC, ties by canonical
  numeric item ID ASC; unseen items get frequency 0 and still enter the
  ranking (full catalog sequence).
- B0 is history-INDEPENDENT: every validation row receives the identical
  full-catalog ranking (no history filtering).
- Evaluation via the shared v2_evaluate layer in STRICT reference mode:
  row_id->(user_id, target_item_id) ground truth parsed from the FROZEN
  valid.csv source (never from predictions); label coverage gate on; test
  never touched.
- --verify reloads all products in a fresh pass and re-derives everything.

No test predictions or metrics anywhere."""
import argparse, csv, json, hashlib, time, ast, sys
from pathlib import Path
from collections import Counter

REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
BASE = REPO / "outputs/v2/baselines/B0"
EV = REPO / "outputs/v2/evidence/phase5"
HEADER = ["row_id", "user_id", "history_item_title", "item_title",
          "history_item_id", "item_id", "history_item_sid", "item_sid"]
KS = (3, 5, 10)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_split_rows(path):
    """Read a phase-4 CSV -> list of dicts {row_id, user_id, target} (original order)."""
    rows = []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        assert hdr == HEADER, f"unexpected header in {path}: {hdr}"
        for r in rd:
            rows.append({"row_id": r[0], "user_id": r[1], "target": int(r[5])})
    return rows


def load_catalog(path):
    idx = json.load(open(path))
    return [int(x) for x in idx["item_ids"]]


def load_labels(path):
    labels = {}
    for line in open(path):
        d = json.loads(line)
        labels[int(d["item_id"])] = bool(d["original_description_missing"])
    return labels


# ---------------- build ----------------

def build(train_csv, catalog_json):
    """Frequency source isolation: ONLY the train file is read here."""
    train_rows = load_split_rows(train_csv)
    freq = Counter(r["target"] for r in train_rows)
    catalog = load_catalog(catalog_json)
    catalog_set = set(catalog)
    assert set(freq.keys()) <= catalog_set, "train target outside catalog (upstream breach)"
    freqs = {i: int(freq.get(i, 0)) for i in catalog}          # complete catalog, 0 for unseen
    ordering = sorted(catalog, key=lambda i: (-freqs[i], i))    # freq DESC, numeric ID ASC
    return {
        "schema": "phase5-popularity-1.0", "date": "2026-11-11", "method": "B0",
        "frequency_source": "outputs/v2/data/A/train.csv item_id column (TRAIN targets only)",
        "n_train_rows": len(train_rows),
        "frequency_sum": sum(freqs.values()),
        "n_catalog": len(catalog),
        "n_unseen_zero_freq": sum(1 for v in freqs.values() if v == 0),
        "tie_rule": "frequency DESC, then canonical numeric item ID ASC",
        "ordering": ordering,
        "frequencies": freqs,
        "inputs_sha256": {"train_csv": sha256_file(train_csv), "catalog_json": sha256_file(catalog_json)},
    }


# ---------------- evaluate ----------------

def evaluate(pop, verify_mode=False):
    sys.path.insert(0, str(REPO / "scripts"))
    from v2_evaluate import validate_and_align, evaluate_rows, score_row
    catalog_json = REPO / "outputs/v2/data/A/Industrial_and_Scientific.index.json"
    valid_csv = REPO / "outputs/v2/data/A/valid.csv"
    qf_json = REPO / "outputs/v2/manifests/quality_features.jsonl"
    catalog = load_catalog(catalog_json)
    labels = load_labels(qf_json)
    ordering = pop["ordering"]

    valid_rows = load_split_rows(valid_csv)
    # reference ground truth from the FROZEN source CSV (independent of predictions)
    reference = {r["row_id"]: {"user_id": r["user_id"], "target_item_id": r["target"]}
                 for r in valid_rows}
    bound_ids = [r["row_id"] for r in valid_rows]
    pred_rows = [{"row_id": r["row_id"], "user_id": r["user_id"],
                  "target_item_id": r["target"], "ranked_item_ids": ordering}
                 for r in valid_rows]
    aligned = validate_and_align(pred_rows, bound_ids, set(catalog), reference)
    metrics = evaluate_rows(aligned, set(catalog), KS, labels)
    return valid_rows, pred_rows, metrics, labels


# ---------------- verify (fresh reload, no reuse of build outputs) ----------------

def verify():
    checks = {}
    pop = json.load(open(BASE / "popularity.json"))
    # 1) rebuild ordering from recorded frequencies + rule
    freqs = pop["frequencies"]
    rebuilt = sorted(freqs.keys(), key=lambda k: (-freqs[k], int(k)))
    checks["ordering_reload_identical"] = rebuilt == [str(i) for i in pop["ordering"]] or \
        rebuilt == pop["ordering"]
    # keys may be str; normalize comparison
    if not checks["ordering_reload_identical"]:
        checks["ordering_reload_identical"] = [int(x) for x in rebuilt] == list(pop["ordering"])
    # 2) frequency sum / completeness / zero count
    checks["frequency_sum_36259"] = sum(freqs.values()) == 36259
    checks["catalog_complete_3686"] = len(freqs) == 3686
    checks["zero_freq_count"] = sum(1 for v in freqs.values() if v == 0)
    # 3) independent frequency recount for 8 sampled items (deterministic sample)
    train_rows = load_split_rows(REPO / "outputs/v2/data/A/train.csv")
    from collections import Counter
    fc = Counter(r["target"] for r in train_rows)
    sample_items = [int(x) for x in list(freqs.keys())[::461][:8]]
    checks["sampled_freq_recount"] = {str(i): {"recorded": freqs[str(i)],
                                               "recounted": int(fc.get(i, 0)),
                                               "match": freqs[str(i)] == int(fc.get(i, 0))}
                                      for i in sample_items}
    # 4) tie-group check: largest tie group ordering correct
    by_freq = {}
    for k, v in freqs.items():
        by_freq.setdefault(v, []).append(int(k))
    tie_sizes = sorted((len(v) for v in by_freq.values() if len(v) > 1), reverse=True)
    top_group = max(by_freq.items(), key=lambda kv: len(kv[1]))
    gids = sorted(top_group[1])
    pos = {i: pop["ordering"].index(i) for i in gids}
    in_order = [pop["ordering"][p] for p in sorted(pos.values())] == gids
    # and the whole ordering is freq-desc non-increasing
    seq = [freqs[str(i)] for i in pop["ordering"]]
    nonincreasing = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    checks["tie_check"] = {"largest_tie_group_size": len(gids), "tie_group_ids_ascending_in_order": in_order,
                           "ordering_freq_nonincreasing": nonincreasing, "all_tie_group_sizes_desc": tie_sizes[:5]}
    # 5) predictions reload: row count / row_id set / first10 == ordering[:10]
    n_pred = 0
    first10_ok = True
    rids = []
    with open(BASE / "valid_predictions.jsonl") as f:
        for line in f:
            d = json.loads(line)
            n_pred += 1
            rids.append(d["row_id"])
            if d["ranked_item_ids"][:10] != pop["ordering"][:10]:
                first10_ok = False
    valid_rows = load_split_rows(REPO / "outputs/v2/data/A/valid.csv")
    checks["predictions_reload"] = {"n_rows": n_pred, "expected": len(valid_rows),
                                    "row_ids_equal_source": rids == [r["row_id"] for r in valid_rows],
                                    "first10_equals_ordering": first10_ok}
    # 6) metrics reload + fresh recompute equality
    sys.path.insert(0, str(REPO / "scripts"))
    from v2_evaluate import validate_and_align, evaluate_rows
    catalog = load_catalog(REPO / "outputs/v2/data/A/Industrial_and_Scientific.index.json")
    labels = load_labels(REPO / "outputs/v2/manifests/quality_features.jsonl")
    reference = {r["row_id"]: {"user_id": r["user_id"], "target_item_id": r["target"]} for r in valid_rows}
    pred_rows = [{"row_id": r["row_id"], "user_id": r["user_id"],
                  "target_item_id": r["target"], "ranked_item_ids": pop["ordering"]}
                 for r in valid_rows]
    aligned = validate_and_align(pred_rows, [r["row_id"] for r in valid_rows], set(catalog), reference)
    m2 = evaluate_rows(aligned, set(catalog), KS, labels)
    m1 = json.load(open(BASE / "valid_metrics.json"))["metrics"]
    def flat(d, pfx=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, pfx + k + "."))
            elif isinstance(v, (int, float, bool)) or v is None:
                out[pfx + k] = v
        return out
    f1, f2 = flat(m1), flat(m2)
    diffs = {k: (f1[k], f2[k]) for k in f1 if f1[k] != f2[k] and
             not (isinstance(f1[k], float) and isinstance(f2[k], float) and abs(f1[k] - f2[k]) < 1e-15)}
    checks["metrics_reload_equal"] = {"n_compared": len(f1), "diffs": diffs}
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "verify"], default="build")
    args = ap.parse_args()
    t0 = time.time()
    if args.mode == "verify":
        checks = verify()
        (EV / "t4_verify_checks.json").write_text(json.dumps(
            {"schema": "phase5-t4-verify-1.0", "date": "2026-11-11", "checks": checks},
            ensure_ascii=False, indent=1))
        print("T4_VERIFY_SHA256:", sha256_file(EV / "t4_verify_checks.json"), flush=True)
        ok = (checks["ordering_reload_identical"] and checks["frequency_sum_36259"]
              and checks["catalog_complete_3686"]
              and all(v["match"] for v in checks["sampled_freq_recount"].values())
              and checks["tie_check"]["tie_group_ids_ascending_in_order"]
              and checks["tie_check"]["ordering_freq_nonincreasing"]
              and checks["predictions_reload"]["n_rows"] == checks["predictions_reload"]["expected"]
              and checks["predictions_reload"]["row_ids_equal_source"]
              and checks["predictions_reload"]["first10_equals_ordering"]
              and not checks["metrics_reload_equal"]["diffs"])
        print("T4_VERIFY_ALL_OK:", ok, flush=True)
        print("T4_VERIFY_DONE", flush=True)
        return

    # ---- build ----
    BASE.mkdir(parents=True, exist_ok=True)
    train_csv = REPO / "outputs/v2/data/A/train.csv"
    catalog_json = REPO / "outputs/v2/data/A/Industrial_and_Scientific.index.json"
    pop = build(train_csv, catalog_json)
    pop["build"] = {"command": ".venv-phase3/bin/python scripts/v2_popularity.py --mode build",
                    "code_sha256": sha256_file(__file__),
                    "started_sec_unrecorded_wall_s": round(time.time() - t0, 1)}
    (BASE / "popularity.json").write_text(json.dumps(pop, ensure_ascii=False, indent=1))

    # ---- validation evaluation (strict reference) + predictions ----
    valid_rows, pred_rows, metrics, labels = evaluate(pop)
    with open(BASE / "valid_predictions.jsonl", "w") as f:
        for r in pred_rows:
            f.write(json.dumps(r) + "\n")
    metrics_out = {"schema": "phase5-b0-valid-metrics-1.0", "date": "2026-11-11",
                   "config": {"popularity_sha256": sha256_file(BASE / "popularity.json"),
                              "predictions_sha256": sha256_file(BASE / "valid_predictions.jsonl"),
                              "valid_csv_sha256": sha256_file(REPO / "outputs/v2/data/A/valid.csv"),
                              "catalog_sha256": sha256_file(REPO / "outputs/v2/data/A/Industrial_and_Scientific.index.json"),
                              "quality_features_sha256": sha256_file(REPO / "outputs/v2/manifests/quality_features.jsonl"),
                              "evaluator_code_sha256": sha256_file(REPO / "scripts/v2_evaluate.py"),
                              "builder_code_sha256": sha256_file(__file__),
                              "reference_mode": True, "n_rows": len(pred_rows),
                              "n_users": metrics["user_macro"]["n_users"],
                              "frequency_source": "train only; valid/test never read by build()"},
                   "metrics": metrics}
    (BASE / "valid_metrics.json").write_text(json.dumps(metrics_out, ensure_ascii=False, indent=1))

    cfg = {"schema": "phase5-b0-config-1.0", "date": "2026-11-11", "method": "B0-popularity",
           "parameters": {"tie_rule": pop["tie_rule"], "ks": list(KS),
                          "frequency_source": pop["frequency_source"], "seed": None,
                          "checkpoint": None, "note": "build-type baseline: no training checkpoint by design; all four products hash-tracked"},
           "inputs_sha256": metrics_out["config"],
           "products_sha256": {p: sha256_file(BASE / p) for p in
                               ("popularity.json", "valid_predictions.jsonl", "valid_metrics.json")},
           "resources": {"wall_build_eval_s": round(time.time() - t0, 1),
                         "device": "CPU-only (no GPU used)",
                         "output_bytes": {p: (BASE / p).stat().st_size for p in
                                          ("popularity.json", "valid_predictions.jsonl", "valid_metrics.json")}},
           "command": ".venv-phase3/bin/python scripts/v2_popularity.py --mode build"}
    (BASE / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    print("POP_frequency_sum:", pop["frequency_sum"], flush=True)
    print("POP_n_catalog:", pop["n_catalog"], "zero_freq:", pop["n_unseen_zero_freq"], flush=True)
    print("POP_top10:", pop["ordering"][:10], flush=True)
    print("EVAL n_rows:", metrics["n_rows"], "users:", metrics["user_macro"]["n_users"], flush=True)
    print("EVAL hr3/5/10:", metrics["hr@3"], metrics["hr@5"], metrics["hr@10"], flush=True)
    print("EVAL ndcg3/5/10:", metrics["ndcg@3"], metrics["ndcg@5"], metrics["ndcg@10"], flush=True)
    print("EVAL macro_ndcg10:", metrics["user_macro"]["ndcg@10"], "macro_hr10:", metrics["user_macro"]["hr@10"], flush=True)
    print("EVAL coverage10:", metrics["coverage@10"], flush=True)
    print("EVAL illegal:", metrics["illegal_rate_global"], "dup3/5/10:", metrics["duplicate_rate@3"], metrics["duplicate_rate@5"], metrics["duplicate_rate@10"], flush=True)
    print("EVAL slices:", json.dumps(metrics["slices"], ensure_ascii=False), flush=True)
    print("CONFIG_SHA256:", sha256_file(BASE / "config.json"), flush=True)
    print("POPULARITY_SHA256:", sha256_file(BASE / "popularity.json"), flush=True)
    print("PREDICTIONS_SHA256:", sha256_file(BASE / "valid_predictions.jsonl"), flush=True)
    print("METRICS_SHA256:", sha256_file(BASE / "valid_metrics.json"), flush=True)
    print("T4_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
