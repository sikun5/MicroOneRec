#!/usr/bin/env python3
"""Append T8-R2/R1 supplement files to artifact_inventory.json (persistence
closeout per boss RESULTS; no model experiments)."""
import hashlib
import json
import os

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
OUT = os.path.join(ROOT, "outputs/v2/evidence/phase6")
CK = os.path.join(ROOT, "outputs/v2/checkpoints")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    inv_path = os.path.join(OUT, "artifact_inventory.json")
    inv = json.load(open(inv_path))
    added = {}
    new_files = [
        ("evidence", "t8_strict_recheck.py", os.path.join(OUT, "t8_strict_recheck.py")),
        ("evidence", "t8_strict_recheck.json", os.path.join(OUT, "t8_strict_recheck.json")),
        ("sources", "t8_formal_raw.py", os.path.join(ROOT, "scripts", "t8_formal_raw.py")),
    ]
    for g in ("A", "B", "C"):
        new_files.append(("formal", f"{g}-SFT/formal_valid_raw_outputs.json",
                          os.path.join(CK, f"{g}-SFT", "formal_valid_raw_outputs.json")))
    for section, rel, p in new_files:
        if os.path.exists(p):
            inv[section][rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
            added[rel] = inv[section][rel]["sha256"]
    inv["r2_supplement"] = {
        "added_files": added,
        "note": ("T8-R1/R2 supplement files appended by boss RESULTS closeout request; "
                 "the rest of this inventory is unchanged from its anchored generation time; "
                 "run_records.md/acceptance.md R2 addendum hashes are anchored in working-record.md"),
        "date": "2026-11-15"}
    with open(inv_path, "w") as f:
        json.dump(inv, f, ensure_ascii=False, indent=1)
    print("INVENTORY_UPDATED")
    print(json.dumps(added, indent=1))


if __name__ == "__main__":
    main()
