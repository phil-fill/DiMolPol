# just a few train functions

import torch
from tqdm import tqdm
from typing import Tuple, Dict

def train_epoch_matrix(model, dataloader, optimizer, device, writer=None, epoch=None, clip_grad=1.0) -> Tuple[Dict, float]:
    """
    train model for one epoch
    """
    model.train()

    totals = dict(loss=0.0, tensor_mae=0.0, trace_mae=0.0, frob_mae=0.0, aniso_mae=0.0)
    n_total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Training]", leave=False)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()

        pred = model(data)  # (B,3,3)
        target = data.molecular_polarization_tensor.view(-1, 3, 3).to(device).float()

        # Metriken (alle liefern Skalar)
        m_tensor = tensor_mae(pred, target)
        m_trace  = trace_mae(pred, target)
        m_frob   = frobenius_mae(pred, target)
        m_aniso  = anisotropic_mae(pred, target)

        # kombinierter Loss
        loss = m_aniso

        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

        # Gewichtet akkumulieren
        num_graphs = data.num_graphs if hasattr(data, "num_graphs") else pred.size(0)
        n_total += num_graphs
        totals["loss"]       += loss.item()    * num_graphs
        totals["tensor_mae"] += m_tensor.item()* num_graphs
        totals["trace_mae"]  += m_trace.item() * num_graphs
        totals["frob_mae"]   += m_frob.item()  * num_graphs
        totals["aniso_mae"]  += m_aniso.item() * num_graphs

        # nur Anzeige in der Progressbar (kein Batch-Logging)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            tMAE=f"{m_tensor.item():.4f}",
            trMAE=f"{m_trace.item():.4f}",
            fMAE=f"{m_frob.item():.4f}",
            aMAE=f"{m_aniso.item():.4f}",
        )

    # Epoch-Mittelwerte
    denom = max(n_total, 1)
    avg = {k: v / denom for k, v in totals.items()}

    # TensorBoard: nur pro Epoche loggen
    if writer is not None and epoch is not None:
        writer.add_scalar("Train/Loss",        avg["loss"],       epoch)
        writer.add_scalar("Train/TensorMAE",   avg["tensor_mae"], epoch)
        writer.add_scalar("Train/TraceMAE",    avg["trace_mae"],  epoch)
        writer.add_scalar("Train/FrobeniusMAE",avg["frob_mae"],   epoch)
        writer.add_scalar("Train/AnisoMAE",    avg["aniso_mae"],  epoch)

    return avg["loss"], avg

@torch.no_grad()
def val_epoch_matrix(model, dataloader, device, writer=None, epoch=None) -> Tuple[Dict, float]:
    """
    validate model  one epoch
    """
    model.eval()

    totals = dict(loss=0.0, tensor_mae=0.0, trace_mae=0.0, frob_mae=0.0, aniso_mae=0.0)
    n_total = 0

    for data in tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False):
        data = data.to(device)
        pred   = model(data)  # (B,3,3)
        target = data.molecular_polarization_tensor.view(-1, 3, 3).to(device).float()

        # Metriken
        m_tensor = tensor_mae(pred, target)
        m_trace  = trace_mae(pred, target)
        m_frob   = frobenius_mae(pred, target)
        m_aniso  = anisotropic_mae(pred, target)

        # kombinierter Loss (Skalar)
        loss = m_aniso

        # gewichtet akkumulieren
        num = data.num_graphs if hasattr(data, "num_graphs") else pred.size(0)
        n_total += num
        totals["loss"]       += loss.item()    * num
        totals["tensor_mae"] += m_tensor.item()* num
        totals["trace_mae"]  += m_trace.item() * num
        totals["frob_mae"]   += m_frob.item()  * num
        totals["aniso_mae"]  += m_aniso.item() * num

    # Epoch-Mittelwerte
    denom = max(n_total, 1)
    avg = {k: v / denom for k, v in totals.items()}

    # TensorBoard: nur pro Epoche
    if writer is not None and epoch is not None:
        writer.add_scalar("Val/Loss", avg["loss"], epoch)
        writer.add_scalar("Val/TensorMAE", avg["tensor_mae"], epoch)
        writer.add_scalar("Val/TraceMAE", avg["trace_mae"], epoch)
        writer.add_scalar("Val/FrobeniusMAE", avg["frob_mae"], epoch)
        writer.add_scalar("Val/AnisoMAE", avg["aniso_mae"], epoch)

    return avg["loss"], avg

@torch.no_grad()
def test_epoch_matrix(model, dataloader, device, writer=None, epoch=None) -> Tuple[Dict, float]:
    """
    test model for one epoch
    """
    model.eval()

    totals = dict(
        loss=0.0, tensor_mae=0.0, trace_mae=0.0, frob_mae=0.0, aniso_mae=0.0,
        gt_frob_mean=0.0, gt_trace_abs_mean=0.0, gt_aniso_mean=0.0, gt_component_abs_mean=0.0,
    )
    n_total = 0

    for data in tqdm(dataloader, desc=f"Epoch {epoch+1} [Test]", leave=False):
        data = data.to(device)
        pred   = model(data)  # (B,3,3)
        target = data.molecular_polarization_tensor.view(-1, 3, 3).to(device).float()

        # --- Metriken ---
        m_tensor = tensor_mae(pred, target)
        m_trace  = trace_mae(pred, target)
        m_frob   = frobenius_mae(pred, target)
        m_aniso  = anisotropic_mae(pred, target)
        loss = (m_tensor + m_trace + m_frob + m_aniso) / 4.0

        # --- Ground-Truth-Skalen (pro Batch) ---
        # Frobenius-Norm von T
        gt_frob = torch.linalg.norm(target, ord='fro', dim=(-2, -1)).mean()

        # |Trace(T)|
        gt_trace_abs = target.diagonal(dim1=-2, dim2=-1).sum(-1).abs().mean()

        # Deviatorische Frobenius-Norm: ||T - (tr/3) I||_F
        I = torch.eye(3, dtype=target.dtype, device=target.device)
        tr = target.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
        dev_t = target - tr[:, None, None] * I
        gt_aniso_abs = dev_t.abs().mean(dim=(-2, -1)).mean()


        # Mittlere |Komponente| über alle 9 Einträge
        gt_comp_abs = target.abs().mean(dim=(-2, -1)).mean()

        # --- Akkumulieren (gewichtet) ---
        num = data.num_graphs if hasattr(data, "num_graphs") else pred.size(0)
        n_total += num

        totals["loss"]       += loss.item()    * num
        totals["tensor_mae"] += m_tensor.item()* num
        totals["trace_mae"]  += m_trace.item() * num
        totals["frob_mae"]   += m_frob.item()  * num
        totals["aniso_mae"]  += m_aniso.item()* num

        totals["gt_frob_mean"]        += gt_frob.item()        * num
        totals["gt_trace_abs_mean"]   += gt_trace_abs.item()   * num
        totals["gt_aniso_mean"]  += gt_aniso_abs.item()  * num
        totals["gt_component_abs_mean"] += gt_comp_abs.item()  * num

    # Epoch-Mittelwerte
    denom = max(n_total, 1)
    avg = {k: v / denom for k, v in totals.items()}

    return avg["loss"], avg

def check_shapes(pred: torch.Tensor, target: torch.Tensor):
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (B,3,3), got pred={tuple(pred.shape)}, target={tuple(target.shape)}")

def frobenius_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    mae of frob norm
    """
    check_shapes(pred, target)
    diff = pred - target                          # (B,3,3)
    frob = torch.linalg.norm(diff, ord='fro', dim=(-2, -1))  # (B,)
    frob_mae = torch.mean(frob) # mae over batch

    return frob_mae

def tensor_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    mae of tensor
    """
    check_shapes(pred, target)

    return (pred - target).abs().mean(dim=(-2, -1)).mean()


def trace_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    mae of tensor trace
    """
    check_shapes(pred, target)
    tr_pred = pred.diagonal(dim1=-2, dim2=-1).sum(-1)   # (B,)
    tr_tgt  = target.diagonal(dim1=-2, dim2=-1).sum(-1) # (B,)
    err = (tr_pred - tr_tgt).abs()                      # (B,)
    return err.mean()

def anisotropic_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    mae off diagonal elements "isotropic"
    """
    I    = torch.eye(3, dtype=pred.dtype, device=pred.device)
    tr_p = pred.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
    tr_t = target.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
    dev_p = pred   - tr_p[:, None, None] * I
    dev_t = target - tr_t[:, None, None] * I
    return (dev_p - dev_t).abs().mean()














