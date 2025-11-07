import torch
import torch.nn as nn
from DiMolPol.DiMolPol.architecture.layers import ScalarChannel, VectorChannel, TensorChannel
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool
from typing import List, Optional, Tuple


def init_weights(m: nn.Module) -> None:
    """Xavier-uniform for Linear weights and zero bias."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def tensor_invariants(t: torch.Tensor) -> torch.Tensor:
    """Return [trace, frob] for a [...,3,3] tensor."""
    trace = t.diagonal(dim1=-2, dim2=-1).sum(-1)
    frob = t.pow(2).sum(dim=(-2, -1)).sqrt()
    return torch.stack([trace, frob], dim=-1)


class GatedChannelReadout(nn.Module):
    """Gate scalar/vector/tensor channels using rotation-invariant norms."""
    def __init__(self, scalar_feature_dim: int, vector_feature_dim: int, tensor_feature_dim: int):
        super().__init__()
        d_in = scalar_feature_dim + vector_feature_dim + 2 * tensor_feature_dim
        d_mid = max(1, d_in // 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_in, d_mid),
            nn.SiLU(),
            nn.Linear(d_mid, tensor_feature_dim),
        )
        self.apply(init_weights)

    def forward(
        self,
        h: torch.Tensor,                   # [N, C_s]
        norms: Optional[torch.Tensor],     # [N, C_s + C_v + 2*C_t]
        t_loc: Optional[torch.Tensor],     # [N, C_t, 3, 3]
    ) -> torch.Tensor:
        alphas = self.gate_mlp(norms)                      # [N, C_t]
        gate = alphas.unsqueeze(-1).unsqueeze(-1)          # [N, C_t, 1, 1]
        return (gate * t_loc).sum(dim=1)                   # [N, 3, 3]


def to_global_frame(t_local: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Map node tensors to global frame via R · T · Rᵀ."""
    return R @ t_local @ R.transpose(-2, -1)


class TensorMessage_Tensors(nn.Module):
    """
    Multi-channel scalar/vector/tensor message passing with gated tensor readout.
    Minimal comments: embeds atom types, stacks per-layer channels, aggregates norms, gates tensors, rotates and pools.
    """
    def __init__(
        self,
        scalar_feature_dims: List[int],
        vector_feature_dims: List[int],
        tensor_feature_dims: List[int],
        z2Embed: dict,
    ):
        super().__init__()
        self.z2Embed = z2Embed
        self.num_atom_types = len(z2Embed)
        self.atom_embedding = nn.Embedding(self.num_atom_types, scalar_feature_dims[0])

        self.scalar_layers = nn.ModuleList()
        self.vector_layers = nn.ModuleList()
        self.tensor_layers = nn.ModuleList()
        for i in range(len(scalar_feature_dims) - 1):
            self.scalar_layers.append(ScalarChannel(scalar_feature_dims[i], scalar_feature_dims[i + 1]))
            self.vector_layers.append(VectorChannel(scalar_feature_dims[i], vector_feature_dims[i + 1], is_first=(i == 0)))
            self.tensor_layers.append(TensorChannel(scalar_feature_dims[i], tensor_feature_dims[i + 1], is_first=(i == 0)))

        self.readout = GatedChannelReadout(scalar_feature_dims[-1], vector_feature_dims[-1], tensor_feature_dims[-1])
        self.apply(init_weights)

    def forward(self, data: Data) -> torch.Tensor:
        x, Z, edge_index, local_frames, batch = data.pos, data.nuclear_charges, data.edge_index, data.local_frames, data.batch
        Zembed = torch.tensor([self.z2Embed[str(int(z.item()))] for z in Z], device=Z.device, dtype=torch.long)

        h = self.atom_embedding(Zembed)     # [N, C_s]
        v: Optional[torch.Tensor] = None    # [N, C_v, 3]
        t: Optional[torch.Tensor] = None    # [N, C_t, 3, 3]

        for s_layer, v_layer, t_layer in zip(self.scalar_layers, self.vector_layers, self.tensor_layers):
            x, h = s_layer(x, h, edge_index, local_frames, batch)
            v = v_layer(x, h, edge_index, local_frames, batch, v)
            t = t_layer(x, h, edge_index, local_frames, batch, t)

        v_norm = v.norm(dim=-1).view(h.size(0), -1)       # [N, C_v]
        t_inv = tensor_invariants(t).view(h.size(0), -1)   # [N, 2*C_t]
        norms = torch.cat([h, v_norm, t_inv], dim=-1)      # [N, C_s+C_v+2*C_t]

        t_local = self.readout(h, norms=norms, t_loc=t)    # [N, 3, 3]
        t_local = 0.5 * (t_local + t_local.transpose(-1, -2))
        t_glob = to_global_frame(t_local, local_frames)

        t_flat = t_glob.view(t_glob.size(0), -1)
        t_pooled_flat = global_add_pool(t_flat, batch)
        return t_pooled_flat.view(-1, 3, 3)


def make_toy_mol(num_atoms: int, z_list: List[int]) -> Data:
    """Tiny random molecule with full directed graph and identity frames."""
    pos = torch.rand(num_atoms, 3) * 2 - 1
    z = torch.tensor(z_list, dtype=torch.float)
    send, recv = torch.meshgrid(torch.arange(num_atoms), torch.arange(num_atoms), indexing="ij")
    mask = send != recv
    edge_index = torch.stack([send[mask], recv[mask]], dim=0)
    local_frames = torch.eye(3).repeat(num_atoms, 1, 1)
    return Data(pos=pos, nuclear_charges=z, edge_index=edge_index, local_frames=local_frames)


if __name__ == "__main__":
    scalar_dims = [128, 128]
    vector_dims = [32, 32]
    tensor_dims = [16, 16]
    z2Embed = {"1": 0, "6": 1, "8": 2}

    model = TensorMessage_Tensors(
        scalar_feature_dims=scalar_dims,
        vector_feature_dims=vector_dims,
        tensor_feature_dims=tensor_dims,
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
