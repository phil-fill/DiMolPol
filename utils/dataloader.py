import json
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import tqdm

from utils.pca_local_frames import compute_frames_torch
from utils.utils import compute_cutoff_edges_torch, compute_knn_edges_torch


class MoleculeDataset(torch.utils.data.Dataset):
    """PyG dataset that loads pickled molecules, builds edges/frames, and yields Data objects."""
    def __init__(self, pickle_path: str, cutoff: int = 0, k_nearest: int = 0, batch_size: int = 32):
        assert (cutoff > 0) != (k_nearest > 0), "Specify either cutoff or k_nearest (not both)."
        self.pickle_path = str(pickle_path)
        self.cutoff = cutoff
        self.k_nearest = k_nearest
        self.batch_size = batch_size
        self.graph_data: List[Data] = self._load_data()
        self.train_subset: Optional[Subset] = None
        self.val_subset: Optional[Subset] = None
        self.test_subset: Optional[Subset] = None

    def __len__(self) -> int:
        """Return number of graphs."""
        return len(self.graph_data)

    def __getitem__(self, idx):
        """Return graph by index or list of graphs for sequence of indices."""
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self.graph_data[int(i)] for i in idx]
        return self.graph_data[int(idx)]

    def get_mol_ids(self) -> List[int]:
        """Return list of molecule_id as Python ints."""
        return [int(d.molecule_id.item()) for d in self.graph_data]

    def _build_edge_index(self, pos: torch.Tensor) -> torch.Tensor:
        """Return [2,E] LongTensor edge_index via cutoff or kNN."""
        if self.cutoff > 0:
            i, j = compute_cutoff_edges_torch(pos, self.cutoff, "undirected")
        else:
            i, j = compute_knn_edges_torch(pos, self.k_nearest, "undirected")
        return torch.stack([i.long(), j.long()], dim=0)

    def _load_data(self) -> List[Data]:
        """Load pickle, construct graphs with edges and local frames."""
        print(f"Loading {self.pickle_path}")
        with open(self.pickle_path, "rb") as f:
            records = pickle.load(f)

        out: List[Data] = []
        for rec in tqdm.tqdm(records):
            pos = torch.tensor(rec["atXYZ"], dtype=torch.float)
            edge_index = self._build_edge_index(pos)
            Z = torch.tensor(rec["atNUM"], dtype=torch.long)
            mP = torch.tensor(rec["mPOL"], dtype=torch.float)
            mT = torch.tensor(rec["mTPOL"], dtype=torch.float)
            aP = torch.tensor(rec["atPOL"], dtype=torch.float)
            mol_id = torch.tensor(int(rec["mol_id"]), dtype=torch.long)
            local_frames = compute_frames_torch(pos, edge_index, Z)

            out.append(
                Data(
                    num_nodes=pos.size(0),
                    pos=pos,
                    edge_index=edge_index,
                    nuclear_charges=Z,
                    molecular_polarizations=mP,
                    molecular_polarization_tensor=mT,
                    atomic_forces=aP,
                    molecule_id=mol_id,
                    conf_id=rec["conf_id"],
                    local_frames=local_frames,
                )
            )
        return out

    def split_by_ratio(self, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
        """Random split into train/val/test by ratios."""
        assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1."
        n = len(self)
        indices = list(range(n))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]
        self.train_subset = Subset(self, train_idx)
        self.val_subset = Subset(self, val_idx)
        self.test_subset = Subset(self, test_idx)
        return train_idx, val_idx, test_idx

    def load_splits_from_json(self, json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load mol_id-based splits from JSON and map to dataset indices."""
        with open(json_path, "r", encoding="utf-8") as f:
            s = json.load(f)

        by_mol = {}
        for idx, mid in enumerate(self.get_mol_ids()):
            by_mol.setdefault(int(mid), []).append(idx)

        def indices_for(mol_list: Sequence[int]) -> np.ndarray:
            idxs, missing = [], []
            for m in mol_list:
                m = int(m)
                if m not in by_mol:
                    missing.append(m)
                    continue
                idxs.extend(by_mol[m])
            if missing:
                raise RuntimeError(f"{len(missing)} mol_id(s) not in dataset (e.g. {missing[:5]})")
            return np.array(idxs, dtype=np.int64)

        train_idx = indices_for(s["train_mols"])
        val_idx = indices_for(s["val_mols"])
        test_idx = indices_for(s["test_mols"])
        self.train_subset = Subset(self, train_idx)
        self.val_subset = Subset(self, val_idx)
        self.test_subset = Subset(self, test_idx)
        return train_idx, val_idx, test_idx

    def get_loaders(
        self,
        batch_size: Optional[int] = None,
        shuffle_train: bool = True,
        drop_last: bool = False,
    ):
        """Return DataLoaders for preset splits."""
        assert self.train_subset is not None and self.val_subset is not None and self.test_subset is not None, \
            "Call split_by_ratio(...) or load_splits_from_json(...) first."
        bs = batch_size or self.batch_size
        train_loader = DataLoader(self.train_subset, batch_size=bs, shuffle=shuffle_train, drop_last=drop_last)
        val_loader = DataLoader(self.val_subset, batch_size=bs, shuffle=False, drop_last=False)
        test_loader = DataLoader(self.test_subset, batch_size=bs, shuffle=False, drop_last=False)
        return train_loader, val_loader, test_loader


def set_seeds(seed: int) -> None:
    """Set Python/NumPy/PyTorch seeds and deterministic cuDNN flags."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
