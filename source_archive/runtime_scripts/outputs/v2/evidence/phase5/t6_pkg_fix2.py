from pathlib import Path
import json, hashlib
EV = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec/outputs/v2/evidence/phase5")
REPO = EV.parents[3]
assert (REPO / "scripts/v2_sasrec.py").is_file(), "REPO resolution wrong"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
p3 = EV / "t6_checks.json"
d = json.load(open(p3))
d["evidence_sha256"] = {
    "validation_metrics.json": sha(EV / "validation_metrics.json"),
    "resource_summary.json": sha(EV / "resource_summary.json"),
    "artifact_inventory.json": "REGENERATED_AFTER_THIS_FIELD_FIX__SEE_RECORD",
    "t6_checks.py": sha(EV / "t6_checks.py"),
}
B = REPO / "outputs/v2/baselines"
inv = {
    "schema": "phase5-artifact-inventory-1.1", "date": "2026-11-12",
    "note": "handover documents (run_records.md/acceptance.md/artifact_inventory.json/t6_checks.json) excluded from self-inclusion; their final hashes are recorded in the shared working record",
    "scripts": {k: sha(REPO / k) for k in (
        "scripts/v2_evaluate.py", "scripts/v2_bootstrap.py", "scripts/v2_generation_adapter.py",
        "scripts/v2_sasrec.py", "scripts/v2_popularity.py")},
    "evidence": {pp.name: sha(pp) for pp in sorted(EV.iterdir())
                 if pp.is_file() and pp.name not in
                 ("run_records.md", "acceptance.md", "artifact_inventory.json", "t6_checks.json")},
    "baselines": {str(pp.relative_to(REPO)): {"sha256": sha(pp), "bytes": pp.stat().st_size}
                  for pp in sorted(B.rglob("*")) if pp.is_file()},
}
(EV / "artifact_inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1))
d["evidence_sha256"]["artifact_inventory.json"] = sha(EV / "artifact_inventory.json")
d["package_reference_fix"] = {
    "date": "2026-11-12",
    "scope": "field/reference-level fix: stale protocol/adapter hashes in acceptance.md updated to post-fix values with historical marks; t6_checks.json evidence_sha256 refreshed; artifact_inventory regenerated (v1.1) excluding self-referential handover docs; NOT independent recompute",
}
p3.write_text(json.dumps(d, ensure_ascii=False, indent=1))
print("t6_checks patched:", sha(p3))
print("FINAL HASHES:")
for f in ("acceptance.md", "run_records.md", "artifact_inventory.json", "t6_checks.json",
          "t6_checks.py", "t6_protocol_check.py", "t6_protocol_check.json",
          "validation_metrics.json", "resource_summary.json", "evaluation_protocol.json"):
    print(f"  {f}={sha(EV / f)}")
print("PKG_FIX_DONE")
