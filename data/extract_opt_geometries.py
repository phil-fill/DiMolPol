#!/usr/bin/env python3
# build_meta_slim_light_xz.py

from pathlib import Path
import pickle, numpy as np, h5py, re, lzma, shutil, tempfile

RAW_DIR = "data/raw_data/"
DUP_MOLS_PATH = "data/raw_data/DupMols.dat"
OUT_PICKLE = "data/meta.pickle"
ATTRIBUTES = ["atNUM", "atXYZ", "atPOL", "mTPOL", "mPOL"]
ONLY_OPT = True  # keep only *-opt conformations

def load_dup_ids(path):
    ids = set()
    p = Path(path)
    if not p.exists():
        return ids
    for line in p.read_text().splitlines():
        m = re.search(r"\d+", line)
        if m:
            ids.add(m.group(0))
    return ids

def as32(x):
    a = np.asarray(x)
    return a.astype(np.float32, copy=False) if a.dtype == np.float64 else a

def decompress_xz(xz_path: Path) -> Path:
    """decompress .xz → .hdf5 in a temp file"""
    tmp = Path(tempfile.gettempdir()) / (xz_path.stem + ".hdf5")
    if not tmp.exists():
        with lzma.open(xz_path, "rb") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return tmp

def main():
    files = sorted(Path(RAW_DIR).glob("*.hdf5")) + sorted(Path(RAW_DIR).glob("*.xz"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 or .xz files in {RAW_DIR}/")

    dup_ids = load_dup_ids(DUP_MOLS_PATH)
    db = []
    skipped = 0

    for f in files:
        # decompress if needed
        path = decompress_xz(f) if f.suffix == ".xz" else f
        with h5py.File(path, "r") as h5:
            for mol_id in sorted(h5.keys(), key=int):
                if mol_id in dup_ids:
                    skipped += 1
                    continue
                for conf_id in h5[mol_id]:
                    if ONLY_OPT and not conf_id.endswith("-opt"):
                        continue
                    conf = h5[mol_id][conf_id]
                    if any(a not in conf for a in ATTRIBUTES):
                        continue
                    rec = {"mol_id": mol_id, "conf_id": conf_id}
                    for a in ATTRIBUTES:
                        rec[a] = as32(conf[a])
                    db.append(rec)

    with open(OUT_PICKLE, "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Files: {len(files)}, skipped: {skipped}, saved: {len(db)} → {OUT_PICKLE}")

if __name__ == "__main__":
    main()
