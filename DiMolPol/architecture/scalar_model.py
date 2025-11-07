import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool
from typing import List, Optional

from DiMolPol.DiMolPol.architecture.layers import ScalarChannel


def init_weights(m: nn.Module) -> None:
    """Initialize Linear layers with Xavier-uniform weights and zero bias."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def tensor_invariants(t: torch.Tensor) -> torch.Tensor:
    """Return trace and Frobenius norm of a [...,3,3] tensor as [...,2]."""
    trace = t.diagonal(dim1=-2, dim2=-1).sum(-1)
    frob = t.pow(2).sum(dim=(-2, -1)).sqrt()
    return torch.stack([trace, frob], dim=-1)


class ScalarModel(nn.Module):
    """Scalar MPNN over atoms that emits graph-level symmetric 3×3 tensors."""
    def __init__(self, scalar_feature_dims: List[int], tensor_feature_dim: int, z2Embed: dict):
        super().__init__()
        self.num_atom_types = len(z2Embed)
        self.atom_embedding = nn.Embedding(self.num_atom_types, scalar_feature_dims[0])
        self.z2Embed = z2Embed
        self.scalar_feature_dims = scalar_feature_dims
        self.tensor_feature_dims = tensor_feature_dim  # reserved for future multi-channel tensors

        # project final scalars into a per-node symmetric 3×3 tensor
        self.tensor_init = nn.Sequential(
            nn.Linear(scalar_feature_dims[-1], scalar_feature_dims[-1] // 2),
            nn.LayerNorm(scalar_feature_dims[-1] // 2),
            nn.SiLU(inplace=True),
            nn.Linear(scalar_feature_dims[-1] // 2, 9),
            nn.LayerNorm(9),
            nn.SiLU(inplace=True),
        )

        # stack scalar channels
        self.scalar_layers = nn.ModuleList(
            [ScalarChannel(scalar_feature_dims[i], scalar_feature_dims[i + 1]) for i in range(len(scalar_feature_dims) - 1)]
        )

        self.apply(init_weights)

    def forward(self, data: Data) -> torch.Tensor:
        """Run scalar message passing, build node tensors, rotate to global, and pool to graph tensors."""
        x, Z, edge_index, local_frames, batch = data.pos, data.nuclear_charges, data.edge_index, data.local_frames, data.batch
        Zembed = torch.tensor([self.z2Embed[str(int(z.item()))] for z in Z], device=Z.device, dtype=torch.long)

        h = self.atom_embedding(Zembed)
        for scalar_layer in self.scalar_layers:
            x, h = scalar_layer(x, h, edge_index, local_frames, batch)

        t_local = self.tensor_init(h).view(h.size(0), 3, 3)
        t_local = 0.5 * (t_local + t_local.transpose(-1, -2))
        t_glob = to_global_frame(t_local, local_frames)

        t_flat = t_glob.view(t_glob.size(0), -1)
        t_pooled_flat = global_add_pool(t_flat, batch)
        t_out = t_pooled_flat.view(-1, 3, 3)
        return t_out


def to_global_frame(t_local: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Map node tensors from local frames to the global frame via R · T · Rᵀ."""
    R_T = R.transpose(-2, -1)
    return torch.matmul(R, torch.matmul(t_local, R_T))


def make_toy_mol(num_atoms: int, z_list: List[int]) -> Data:
    """Create a tiny random molecule with full directed graph and identity local frames."""
    pos = torch.rand(num_atoms, 3) * 2 - 1
    z = torch.tensor(z_list, dtype=torch.float)
    send, recv = torch.meshgrid(torch.arange(num_atoms), torch.arange(num_atoms), indexing="ij")
    mask = send != recv
    edge_index = torch.stack([send[mask], recv[mask]], dim=0)
    local_frames = torch.eye(3).repeat(num_atoms, 1, 1)
    return Data(pos=pos, nuclear_charges=z, edge_index=edge_index, local_frames=local_frames)


if __name__ == "__main__":
    scalar_dims = [128, 128]
    z2Embed = {"1": 0, "6": 1, "8": 2}

    model = ScalarModel(
        scalar_feature_dims=scalar_dims,
        tensor_feature_dim=1,
        z2Embed=z2Embed,
    ).eval()

    mol1 = make_toy_mol(3, [1, 1, 8])
    mol2 = make_toy_mol(4, [6, 1, 1, 1])
    mol3 = make_toy_mol(2, [8, 8])
    batch = Batch.from_data_list([mol1, mol2, mol3])

    with torch.no_grad():
        mol_tensors = model(batch)

    for i, T in enumerate(mol_tensors, start=1):
        print(f"\nMolecule {i} – global 3x3 tensor:")
        print(T)
