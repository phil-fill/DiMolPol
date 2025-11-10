import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ScalarChannel(nn.Module):
    """
    Message-passing on scalar features with gated messages and optional coordinate updates.
    Uses edge features (r^2, 1/(r^2+eps)) and aggregates with scatter-add.
    Two independent gates: one for message strength, one for coordinate update weights.
    """
    def __init__(self, in_channels: int, out_channels: int, use_coord_update: bool = False):
        super().__init__()
        self.use_coord_update = use_coord_update

        self.phi_e = nn.Sequential(
            nn.Linear(in_channels * 2 + 2, out_channels),
            nn.LayerNorm(out_channels), nn.SiLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels), nn.SiLU(inplace=True),
        )

        self.msg_gate = nn.Sequential(
            nn.Linear(out_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.phi_h = nn.Sequential(
            nn.Linear(out_channels + in_channels, out_channels),
            nn.LayerNorm(out_channels), nn.SiLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
        )

        self.coord_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels), nn.SiLU(inplace=True),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )

        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        local_frames: torch.Tensor,  # unused here but kept for API consistency
        batch: torch.Tensor,         # unused here but kept for API consistency
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        r_ij = x[col] - x[row]
        r2 = (r_ij ** 2).sum(dim=-1, keepdim=True)
        pot = 1.0 / (r2 + 1e-6)

        hi, hj = h[col], h[row]
        edge_in = torch.cat([hi, hj, r2, pot], dim=-1)
        m_ij = self.phi_e(edge_in)

        g_msg = self.msg_gate(m_ij)
        m_ij = m_ij * g_msg

        m_i = torch.zeros_like(h).scatter_add_(0, col.unsqueeze(-1).expand(-1, m_ij.size(-1)), m_ij)
        h = h + self.phi_h(torch.cat([h, m_i], dim=-1))

        if self.use_coord_update:
            w_ij = self.coord_gate(m_ij)
            dx = r_ij * w_ij
            x = x + torch.zeros_like(x).index_add_(0, row, dx)

        return x, h


class VectorChannel(nn.Module):
    """
    Vector features per node updated by gated interactions in local frames.
    Initializes vectors from scalars; rotates sender vectors into receiver frames; aggregates with learned mixing.
    """
    def __init__(self, in_scalar_dim: int, vector_dim: int, is_first: bool = False):
        super().__init__()
        self.vector_dim = vector_dim
        self.is_first = is_first

        self.vector_init = nn.Sequential(
            nn.Linear(in_scalar_dim, vector_dim * 3),
            nn.LayerNorm(vector_dim * 3),
            nn.SiLU(inplace=True),
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.LayerNorm(vector_dim * 3),
            nn.SiLU(inplace=True),
        )

        self.vector_interaction = nn.Sequential(
            nn.Linear(in_scalar_dim * 2, vector_dim * (vector_dim * 2)),
            nn.LayerNorm(vector_dim * (vector_dim * 2)),
            nn.Sigmoid(),
        )

        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        local_frames: torch.Tensor,
        batch: torch.Tensor,                 # unused here but kept for API consistency
        v_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        row, col = edge_index
        N = x.size(0)

        vectors = self.vector_init(h).view(-1, self.vector_dim, 3)

        hi, hj = h[col], h[row]
        vi, vj = vectors[col], vectors[row]
        Ri, Rj = local_frames[col], local_frames[row]

        R_rel = torch.bmm(Ri.transpose(1, 2), Rj)
        vj_in_i = torch.einsum("eij,ecj->eci", R_rel, vj)

        stacked_vectors = torch.cat([vi, vj_in_i], dim=1)
        stacked_norms = torch.cat([hi, hj], dim=-1)
        int_weights_ij = self.vector_interaction(stacked_norms).view(-1, self.vector_dim * 2, self.vector_dim)

        v_new = torch.einsum("eko,ekc->eco", stacked_vectors, int_weights_ij)

        v_out_loc = torch.zeros((N, self.vector_dim, 3), device=v_new.device, dtype=v_new.dtype)
        v_out_loc = v_out_loc.scatter_add(
            dim=0,
            index=col.view(-1, 1, 1).expand(-1, self.vector_dim, 3),
            src=v_new,
        )

        return v_out_loc if v_prev is None else v_prev + v_out_loc


class TensorChannel(nn.Module):
    """
    Rank-2 tensor features per node updated via frame-aligned sender tensors.
    Initializes tensors from scalars, rotates sender with R_rel * T * R_rel^T, mixes with learned weights, symmetrizes.
    """
    def __init__(self, in_dim: int, tensor_dim: int, is_first: bool):
        super().__init__()
        self.tensor_dim = tensor_dim
        self.is_first = is_first

        self.tensor_init = nn.Sequential(
            nn.Linear(in_dim, tensor_dim * 9),
            nn.LayerNorm(tensor_dim * 9),
            nn.SiLU(inplace=True),
            nn.Linear(tensor_dim * 9, tensor_dim * 9),
            nn.LayerNorm(tensor_dim * 9),
            nn.SiLU(inplace=True),
        )

        self.tensor_interaction = nn.Sequential(
            nn.Linear(in_dim * 2, tensor_dim * (tensor_dim * 2)),
            nn.LayerNorm(tensor_dim * (tensor_dim * 2)),
            nn.Sigmoid(),
        )

        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        local_frames: torch.Tensor,
        batch: torch.Tensor,
        t_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = x.size(0)
        row, col = edge_index

        hi, hj = h[col], h[row]
        tensors = self.tensor_init(h).view(-1, self.tensor_dim, 3, 3)

        ti, tj = tensors[col], tensors[row]
        Ri, Rj = local_frames[col], local_frames[row]

        R_rel = torch.bmm(Ri.transpose(1, 2), Rj)
        tj_in_i = torch.einsum("eik,eckl,ejl->ecij", R_rel, tj, R_rel.transpose(1, 2))

        stacked_tensor = torch.cat([ti, tj_in_i], dim=1)
        stacked_scalars = torch.cat([hi, hj], dim=-1)
        int_weights_ij = self.tensor_interaction(stacked_scalars).view(-1, self.tensor_dim * 2, self.tensor_dim)

        t_new = torch.einsum("ekij,ekc->ecij", stacked_tensor, int_weights_ij)

        t_out_loc = t_new.new_zeros(N, self.tensor_dim, 3, 3)
        index = col.view(-1, 1, 1, 1).expand(-1, self.tensor_dim, 3, 3)
        t_out_loc.scatter_add_(0, index, t_new)

        t_out_loc = 0.5 * (t_out_loc + t_out_loc.transpose(-1, -2))

        return t_out_loc if t_prev is None else t_prev + t_out_loc
