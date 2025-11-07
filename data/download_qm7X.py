# load qm7-X data

import os
import subprocess

os.makedirs("data/raw_data", exist_ok=True)

files = [
    "1000.xz",
    "2000.xz",
    "3000.xz",
    "4000.xz",
    "5000.xz",
    "6000.xz",
    "7000.xz",
    "8000.xz",
    "createDB.py",
    "DupMols.dat",
    "README.txt",
]

for f in files:
    url = f"https://zenodo.org/records/4288677/files/{f}?download=1"
    dest = os.path.join("data/raw_data", f)
    print(f"Lade {f} ...")
    subprocess.run(["wget", "-c", url, "-O", dest])
