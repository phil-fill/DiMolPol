#!/usr/bin/env python3
# reproducable splits

import json, pickle, random
from pathlib import Path

META_PATH = "data/meta.pickle"
OUT_JSON = "data/splits_mol.json"

TRAIN_R = 0.8
VAL_R = 0.1
SEED = 42

def largest_remainder_counts(n, ratios):
    """integer counts via largest remainder method"""
    targets = [r * n for r in ratios]
    floors = [int(t) for t in targets]
    rest = n - sum(floors)
    order = sorted(range(len(ratios)), key=lambda i: targets[i] - floors[i], reverse=True)
    for i in order[:rest]:
        floors[i] += 1
    return floors

def main():
    p = Path(META_PATH)
    if not p.exists():
        raise FileNotFoundError(f"{META_PATH} not found")

    with open(p, "rb") as f:
        db = pickle.load(f)
    if not db:
        raise RuntimeError("meta.pickle empty")

    # group by molecule id
    by_mol = {}
    for rec in db:
        mid = int(rec["mol_id"])
        by_mol.setdefault(mid, []).append(rec)

    mols = list(by_mol.keys())
    if len(mols) < 3:
        raise RuntimeError("too few molecules")

    rnd = random.Random(SEED)
    rnd.shuffle(mols)

    r_test = 1.0 - TRAIN_R - VAL_R
    n_train, n_val, n_test = largest_remainder_counts(len(mols), [TRAIN_R, VAL_R, r_test])

    train = mols[:n_train]
    val = mols[n_train:n_train+n_val]
    test = mols[n_train+n_val:]

    data = {"seed": SEED, "train_mols": train, "val_mols": val, "test_mols": test}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Train {len(train)} | Val {len(val)} | Test {len(test)} | Total {len(mols)}")
    print(f"Saved → {OUT_JSON}")

if __name__ == "__main__":
    main()
