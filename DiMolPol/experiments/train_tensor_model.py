#!/usr/bin/env python3
# train_qm7x.py

import random
from collections import defaultdict
from pathlib import Path
import json
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset


from DiMolPol.architecture.tensor_model import TensorMessage_Tensors
from DiMolPol.experiments.train_functions import *
from utils.constants import Z2EMBED_QM7X
from utils.dataloader import MoleculeDataset, set_seeds

from pathlib import Path
print("CWD:", Path.cwd())


META_PATH = "data/meta.pickle"


# Splits:
SPLIT_JSON    = "data/splits_mol.json"

EPOCHS               = 1000
BATCH_SIZE           = 32
START_LR             = 1e-4
NUM_FEATURES_SCALARS = 128
NUM_FEATURES_VECTORS = 4
NUM_FEATURES_TENSORS = 32
NUM_LAYERS           = 8
CUTOFF_RADIUS        = 4
SEED                 = 42




def run():
    set_seeds(SEED)

    # --- CUDA-Info sicher ausgeben
    if torch.cuda.is_available():
        print("✅ CUDA available!")
        try:
            print("CUDA version:", torch.version.cuda)
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("[WARN] CUDA-Info:", e)
    else:
        print("❌ CUDA not available")

    # load dataset
    ds = MoleculeDataset(META_PATH, cutoff=CUTOFF_RADIUS, batch_size=BATCH_SIZE)
    print(f"[INFO] loaded data {len(ds)}")

    # create dataset with splits
    train_idx, val_idx, test_idx = ds.load_splits_from_json(SPLIT_JSON)
    print(f"[SPLIT] Configs  -> Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")

    # load data
    train_set = Subset(ds, train_idx)
    val_set = Subset(ds, val_idx)
    test_set = Subset(ds, test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    SCALAR_LAYERS = [NUM_FEATURES_SCALARS] * NUM_LAYERS
    VECTOR_LAYERS = [NUM_FEATURES_VECTORS] * NUM_LAYERS
    TENSOR_LAYERS = [NUM_FEATURES_TENSORS] * NUM_LAYERS

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_name = (
        f"Tensor_Cutoff_{CUTOFF_RADIUS}"
        f"_LR_{START_LR}"
        f"_Batch_{BATCH_SIZE}"
        f"_Layers{NUM_LAYERS}"
        f"_Scalar{NUM_FEATURES_SCALARS}"
        f"_Vector{NUM_FEATURES_VECTORS}"
        f"_Tensor{NUM_FEATURES_TENSORS}"
    )
    ckpt_dir = Path("DiMolPol/experiments/checkpoints") / config_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = ckpt_dir / "best_model.pt"  # nur Gewichte
    best_full_path = ckpt_dir / "best_full.ckpt"  # optional: voller Checkpoint

    writer = SummaryWriter(log_dir=f"DiMolPol/experiments/logs/{config_name}")

    model = TensorMessage_Tensors(
        SCALAR_LAYERS, VECTOR_LAYERS, TENSOR_LAYERS, Z2EMBED_QM7X
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=START_LR, weight_decay=1e-16)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # train loop
    for epoch in range(EPOCHS):
        train_loss, train_metrics = train_epoch_matrix(model, train_loader, optimizer, DEVICE, writer=writer, epoch=epoch)

        print(f"[Train][Epoch {epoch + 1}] "
              f"Loss: {train_loss:.6f} | "
              f"TensorMAE: {train_metrics['tensor_mae']:.6f} | "
              f"TraceMAE: {train_metrics['trace_mae']:.6f} | "
              f"FrobeniusMAE: {train_metrics['frob_mae']:.6f} | "
              f"AnisoMAE: {train_metrics['aniso_mae']:.6f}")



        # Validation
        val_loss, val_metrics = val_epoch_matrix(model, val_loader, DEVICE, writer, epoch)

        print(f"[Val][Epoch {epoch + 1}] "
              f"Loss: {val_loss:.6f} | "
              f"TensorMAE: {val_metrics['tensor_mae']:.6f} | "
              f"TraceMAE: {val_metrics['trace_mae']:.6f} | "
              f"FrobeniusMAE: {val_metrics['frob_mae']:.6f} | "
              f"AnisoMAE: {val_metrics['aniso_mae']:.6f}")

        if val_loss < best_val:
            best_val = float(val_loss)
            torch.save(model.state_dict(), best_path)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": best_val,
                "config_name": config_name,
            }, best_full_path)
            print(f"[Checkpoint] New best val loss {best_val:.6f} at epoch {epoch + 1} → {best_path}")



        scheduler.step()

if __name__ == "__main__":
    run()
