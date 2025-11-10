# DiMolPol – Getting Started

This project provides scalar and tensor neural-network models for learning molecular polarizabilities on the QM7-X dataset. The steps below walk you through setting up the environment, preparing the QM7-X data, and starting a training run **from the repository root** (e.g. `/home/user/projects/DiMolPol/`).

## Prerequisites
- Conda (Miniconda/Anaconda). Python 3.9–3.11 works best; PyTorch Geometric had some issues for me with 3.12 on some GPUs (e.g. RTX 4090).
- `wget` (used by the downloader).
- PyTorch, PyTorch Geometric, NumPy, h5py, and TensorBoard. Create and populate a Conda environment once:

```bash
conda create -n dimolpol python=3.11 -y
conda activate dimolpol
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # choose wheel matching your CUDA/CPU setup
pip install torch-geometric numpy h5py tensorboard
```

## 1. Download QM7-X
Run the downloader once to fetch the compressed QM7-X shards (several GB). The files land in `data/raw_data/`.

```bash
python3 data/download_qm7X.py
```

The script resumes partial downloads thanks to `wget -c`. Make sure you have enough free disk space before starting.

## 2. Build `meta.pickle`
Extract the optimized conformers and serialize them into a single meta file.

```bash
python3 data/extract_opt_geometries.py
```

This script:
- Decompresses each `.xz` shard to a temporary `.hdf5`.
- Drops duplicate molecule IDs listed in `data/raw_data/DupMols.dat`.
- Keeps only `*-opt` conformations and the attributes defined in the script header.
- Writes the final dataset to `data/meta.pickle`.

If the script cannot find any `.xz/.hdf5` files it will raise `FileNotFoundError`, which usually means the download step was skipped or pointed to the wrong directory.

## 3. Create train/val/test splits
Generate reproducible molecule-level splits (80 / 10 / 10 by default).

```bash
python3 data/make_splits.py
```

Successful execution saves `data/splits_mol.json`, which the training scripts read to construct `Subset` objects.

## 4. Train a model
All training commands must also be executed from the project root so module imports resolve correctly.

Train the tensor message-passing model:
```bash
python3 -m DiMolPol.experiments.train_tensor_model
```

Train the scalar baseline:
```bash
python3 -m DiMolPol.experiments.train_scalar_model
```

Both scripts:
- Expect `data/meta.pickle` and `data/splits_mol.json` to exist.
- Automatically detect CUDA, fall back to CPU otherwise.
- Log TensorBoard summaries to `DiMolPol/experiments/logs/<config_name>/`.
- Save checkpoints to `DiMolPol/experiments/checkpoints/<config_name>/` (best weights in `best_model.pt` and an optional full checkpoint in `best_full.ckpt`).

## Tips & Troubleshooting
- Ensure `pwd` prints the repository root before running any command; relative paths are hard-coded.
- If downloads fail, verify your network allows `https://zenodo.org` and re-run `download_qm7X.py`; partial files will resume.
- Delete or move old checkpoints if you change hyperparameters but reuse the same configuration name, otherwise the script will overwrite previous results.
