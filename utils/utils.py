import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, List
import torch
import torch
import numpy as np

def _normalize_torch(v: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    normalize torch tensor
    """
    n = torch.linalg.norm(v, dim=dim, keepdim=True).clamp_min(eps)
    return v / n

@torch.no_grad()
def compute_frames_torch(
    pos: torch.Tensor,           # [N,3] float
    edge_index,                  # [2,E] long *oder* Tuple(Tensor,Tensor)
    charges: torch.Tensor | None = None,
    weight_mode: str = "charge",   # {"dist","charge","ones"}
    eps: float = 1e-8
) -> torch.Tensor:              # [B,N,3,3]
    """SO(3)-equivariant frames
    """
    # normalize edge_index layout
    if isinstance(edge_index, (list, tuple)):
        edge_index = torch.stack(edge_index, dim=0)
    elif isinstance(edge_index, torch.Tensor):
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be [2,E]")
    else:
        raise TypeError("edge_index must be Tensor or (row,col) sequence")

    N = pos.size(0)
    device = pos.device
    dtype = pos.dtype
    row, col = edge_index.to(device=device, dtype=torch.long)

    d_ij = pos[row] - pos[col]                              # [E,3]  (x_j - x_i)
    dist = torch.linalg.norm(d_ij, dim=-1).clamp_min(eps)   # [E]

    if weight_mode == "dist":
        w = (1.0 / dist).clamp_max(1e3)
    elif weight_mode == "charge":
        if charges is None:
            raise ValueError("charges=None, aber weight_mode='charge' gewählt.")
        w = charges[row].to(dtype=dtype).abs().clamp_min(eps)
    elif weight_mode == "ones":
        w = torch.ones_like(dist)
    else:
        raise ValueError(f"Unbekannter weight_mode: {weight_mode}")

    # sums per target node i (col)
    sum_w = torch.bincount(col, weights=w, minlength=N).to(dtype=dtype).clamp_min(eps)
    mu_num = torch.zeros((N, 3), device=device, dtype=dtype)
    mu_num.index_add_(0, col, w.unsqueeze(-1) * d_ij)
    mu = mu_num / sum_w.unsqueeze(-1)
    x_raw = mu.clone()

    outer_edges = d_ij.unsqueeze(-1) * d_ij.unsqueeze(-2)
    weighted_outer = w.view(-1, 1, 1) * outer_edges
    M2 = torch.zeros((N, 3, 3), device=device, dtype=dtype)
    M2.index_add_(0, col, weighted_outer)
    C = (M2 / sum_w.view(-1, 1, 1)) - (mu.unsqueeze(-1) @ mu.unsqueeze(-2))

    evals, evecs = torch.linalg.eigh(C)                     # aufsteigend
    z_pca = _normalize_torch(evecs[..., :, 0], dim=-1, eps=eps)  # kleinste Varianz
    e1    = _normalize_torch(evecs[..., :, 2], dim=-1, eps=eps)  # größte Varianz

    dot_zF = (z_pca * mu_num).sum(dim=-1, keepdim=True)
    sgn_z = torch.sign(dot_zF)
    sgn_z[sgn_z == 0.0] = 1.0
    z = z_pca * sgn_z

    mu_norm = torch.linalg.norm(x_raw, dim=-1)
    has_dir = mu_norm > 1e-12
    x_raw = torch.where(has_dir.unsqueeze(-1), x_raw, e1)

    proj_coeff = (x_raw * z).sum(dim=-1, keepdim=True)
    x_tan = x_raw - proj_coeff * z
    x_norm = torch.linalg.norm(x_tan, dim=-1, keepdim=True)
    near_zero_x = (x_norm.squeeze(-1) <= 1e-12)
    if near_zero_x.any():
        x_alt = e1 - (e1 * z).sum(dim=-1, keepdim=True) * z
        x_tan = torch.where(near_zero_x.unsqueeze(-1), x_alt, x_tan)
        x_norm = torch.linalg.norm(x_tan, dim=-1, keepdim=True)
    x = x_tan / x_norm.clamp_min(eps)

    y = _normalize_torch(torch.cross(z, x, dim=-1), dim=-1, eps=eps)
    z = _normalize_torch(torch.cross(x, y, dim=-1), dim=-1, eps=eps)

    R = torch.stack([x, y, z], dim=-1)                        # [N,3,3]
    return R



def compute_cutoff_edges(points: np.ndarray, cutoff_radius: float, direction: str = "undirected"):
    """
    cutoff edges in numpy
    """
    tree = cKDTree(points)
    pairs = tree.query_pairs(cutoff_radius)
    start_indices, end_indices = zip(*pairs) if pairs else ([], [])
    if direction == "directed":
        return np.array(start_indices), np.array(end_indices)
    if direction == "undirected":
        start = np.concatenate((np.array(start_indices), np.array(end_indices)), axis=0)
        end   = np.concatenate((np.array(end_indices), np.array(start_indices)), axis=0)
        return start, end

def compute_knn_edges(positions: np.ndarray, k: int = 5, direction: str="undirected"):
    """
    knn edges numpy
    """
    tree = cKDTree(positions)
    start_indices = []; end_indices = []
    for i, pos in enumerate(positions):
        _, indices = tree.query(pos, k=k + 1)  # +1 wegen Self
        for j in indices[1:]:
            start_indices.append(i); end_indices.append(j)
    if direction == "directed":
        return np.array(start_indices), np.array(end_indices)
    if direction == "undirected":
        start = np.concatenate((np.array(start_indices), np.array(end_indices)), axis=0)
        end   = np.concatenate((np.array(end_indices), np.array(start_indices)), axis=0)
        return start, end


def compute_cutoff_edges_torch(points: torch.Tensor, cutoff_radius: float, direction: str = "undirected"):
    """
    cutoff edges in torch
    """
    dist = torch.cdist(points, points)
    dist.fill_diagonal_(float('inf'))
    N = points.size(0)
    tril = torch.triu(torch.ones((N, N), dtype=torch.bool, device=points.device), diagonal=1)
    mask = (dist <= cutoff_radius) & tril
    idx = mask.nonzero(as_tuple=False)
    i_idx, j_idx = idx[:, 0], idx[:, 1]
    if direction == "directed":
        return i_idx, j_idx
    start = torch.cat([i_idx, j_idx], dim=0)
    end   = torch.cat([j_idx, i_idx], dim=0)
    return start, end

def compute_knn_edges_torch(positions: torch.Tensor, k: int = 5, direction: str = "undirected"):
    """
    knn edges in torch
    """
    N = positions.size(0)
    dist = torch.cdist(positions, positions)
    dist.fill_diagonal_(float('inf'))
    _, idx = torch.topk(dist, k, largest=False)
    i_idx = torch.arange(N, device=positions.device).unsqueeze(1).expand(-1, k).reshape(-1)
    j_idx = idx.reshape(-1)
    if direction == "directed":
        return i_idx, j_idx
    start = torch.cat([i_idx, j_idx], dim=0)
    end   = torch.cat([j_idx, i_idx], dim=0)
    return start, end



