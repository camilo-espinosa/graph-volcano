import gc
import json
from pathlib import Path
import time
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler, Dataset

from . import data_utils
from .station_info import get_default_volcano_to_index, infer_volcano_name_from_path
from utils.model_registry import get_model_spec
from models.UNet_GraphSAGE import UNet_GraphSAGE
from models.UNet_MPNN import UNet_MPNN


def dice_loss_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    class_weights: torch.Tensor = None,
):
    """Legacy 2D Dice loss used in original training code (flattened global Dice)."""
    smooth = 1.0
    iflat = pred.contiguous().view(-1)
    iflat = iflat / iflat.max()
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    a_sum = torch.sum(iflat)
    b_sum = torch.sum(tflat)
    return 1 - ((2.0 * intersection + smooth) / (a_sum + b_sum + smooth))


def dice_loss_graphsage(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    class_weights: Optional[torch.Tensor] = None,
):
    """Dice loss for GraphSAGE outputs [B,C,T] or [B,S,C,T]."""
    _ = smooth, class_weights
    if pred.ndim == 4:
        pred = torch.softmax(pred, dim=2)
        target = target.unsqueeze(1).expand(-1, pred.shape[1], -1, -1)
    elif pred.ndim == 3:
        pred = torch.softmax(pred, dim=1)
    else:
        raise ValueError(
            f"Unexpected pred shape {tuple(pred.shape)}; expected [B,C,T] or [B,S,C,T]."
        )

    iflat = pred.contiguous().view(-1)
    iflat = iflat / iflat.max()
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    a_sum = torch.sum(iflat)
    b_sum = torch.sum(tflat)
    return 1 - ((2.0 * intersection + 1.0) / (a_sum + b_sum + 1.0))


def classwise_dice_graphsage(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Class-wise Dice scores for GraphSAGE outputs [B,C,T] or [B,S,C,T]."""
    if pred.ndim == 4:
        pred = torch.softmax(pred, dim=2).mean(dim=1)
    elif pred.ndim == 3:
        pred = torch.softmax(pred, dim=1)
    else:
        raise ValueError(
            f"Unexpected pred shape {tuple(pred.shape)}; expected [B,C,T] or [B,S,C,T]."
        )

    if target.ndim != 3:
        raise ValueError(
            f"Unexpected target shape {tuple(target.shape)}; expected [B,C,T]."
        )

    target = target.float()
    pred = pred.float()
    intersection = (pred * target).sum(dim=(0, 2))
    pred_sum = pred.sum(dim=(0, 2))
    target_sum = target.sum(dim=(0, 2))
    return (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)


def combined_dice_ce_loss(
    pred: torch.Tensor,
    target_onehot: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    dice_weight: float = 0.7,
    ce_weight: float = 0.3,
):
    """Weighted sum of class-wise Dice loss and CrossEntropy."""
    dice = classwise_dice_graphsage(pred, target_onehot, smooth=1e-6)
    target_idx = torch.argmax(target_onehot, dim=1).long()
    ce = torch.nn.functional.cross_entropy(
        pred,
        target_idx,
        weight=class_weights,
    )
    dice_loss = (1.0 - dice).mean()
    total = dice_weight * dice_loss + ce_weight * ce
    return total, dice_loss, ce


def combined_dice_ce_loss_2d(
    pred: torch.Tensor,
    target_onehot: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    dice_weight: float = 0.7,
    ce_weight: float = 0.3,
):
    """Weighted class-wise Dice + CrossEntropy for 2D UNet logits [B,C,H,W]."""
    if pred.ndim != 4:
        raise ValueError(
            f"Unexpected pred shape {tuple(pred.shape)}; expected [B,C,H,W]."
        )
    if target_onehot.ndim != 4:
        raise ValueError(
            f"Unexpected target shape {tuple(target_onehot.shape)}; expected [B,C,H,W]."
        )

    pred_probs = torch.softmax(pred, dim=1)
    target_onehot = target_onehot.float()
    pred_probs = pred_probs.float()

    intersection = (pred_probs * target_onehot).sum(dim=(0, 2, 3))
    pred_sum = pred_probs.sum(dim=(0, 2, 3))
    target_sum = target_onehot.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)

    target_idx = torch.argmax(target_onehot, dim=1).long()
    ce = torch.nn.functional.cross_entropy(
        pred,
        target_idx,
        weight=class_weights,
    )
    dice_loss = (1.0 - dice).mean()
    total = dice_weight * dice_loss + ce_weight * ce
    return total, dice_loss, ce


def extract_descriptor_tensor(
    descriptor_payload,
    descriptor_names: Sequence[str],
    xb_tensor: torch.Tensor,
) -> torch.Tensor:
    """Build descriptor tensor [B, D, S, Tdesc] from payload keys in fixed order."""
    if descriptor_payload is None:
        raise ValueError("Descriptor payload is missing from batch.")

    desc_list = []
    for name in descriptor_names:
        if name not in descriptor_payload:
            raise ValueError(f"Descriptor '{name}' not found in batch payload.")

        desc = descriptor_payload[name]
        if not torch.is_tensor(desc):
            desc = torch.as_tensor(desc)

        desc = desc.to(device=xb_tensor.device, dtype=xb_tensor.dtype)

        if desc.ndim == 3:
            desc_bst = desc
        elif desc.ndim == 4 and desc.shape[1] == 1:
            desc_bst = desc[:, 0, :, :]
        elif desc.ndim == 4 and desc.shape[2] == 1:
            desc_bst = desc[:, :, 0, :]
        else:
            raise ValueError(
                f"Unsupported descriptor shape for '{name}': {tuple(desc.shape)}."
            )

        if (
            desc_bst.shape[0] != xb_tensor.shape[0]
            or desc_bst.shape[1] != xb_tensor.shape[1]
        ):
            raise ValueError(
                f"Descriptor '{name}' shape {tuple(desc_bst.shape)} incompatible with waveform shape {tuple(xb_tensor.shape)}."
            )
        desc_list.append(desc_bst)

    return torch.stack(desc_list, dim=1)


def save_confusion_matrix_image(
    cm: np.ndarray,
    labels: list,
    out_path: Path,
    title: str,
):
    """Save confusion matrix with count + row-wise percentage annotations."""
    cm_counts = np.asarray(cm)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_pct = (
        np.divide(
            cm_counts.astype(np.float32),
            row_sums,
            out=np.zeros_like(cm_counts, dtype=np.float32),
            where=row_sums > 0,
        )
        * 100.0
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percentage (%)")

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_pct.max() / 2.0 if cm_pct.size > 0 else 0.0
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            count_val = int(cm_counts[i, j])
            pct_val = float(cm_pct[i, j])
            ax.text(
                j,
                i,
                f"{count_val}\n{pct_val:.1f}%",
                ha="center",
                va="center",
                color="white" if pct_val > thresh else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def longest_event(bg_diff: np.ndarray):
    start_indices = np.where(bg_diff == -1)[0]
    if len(start_indices) == 0:
        start_indices = np.array([0])
    end_indices = np.where(bg_diff == 1)[0]
    if len(end_indices) == 0:
        end_indices = np.array([-1])
    events = []
    last_end_idx = -1
    for start in start_indices:
        valid_ends = end_indices[end_indices > start]
        if valid_ends.size > 0:
            end = valid_ends[0]
            events.append((start, end, end - start))
            last_end_idx = end
        else:
            events.append([start, len(bg_diff) - 1, len(bg_diff) - 1 - start])
    if last_end_idx != -1:
        invalid_ends = end_indices[end_indices < start_indices[0]]
        for invalid_end in invalid_ends:
            events.insert(0, (0, invalid_end, invalid_end))
    events_df = pd.DataFrame(events, columns=["start", "end", "length"])
    idx_max = events_df["length"].idxmax()
    start_ = events_df["start"][idx_max]
    end_ = events_df["end"][idx_max]
    length_ = events_df["length"][idx_max]
    return start_, end_, length_


def predicted_from_output(
    out_np,
    clases_ovdas={
        1.0: "VT",
        2.0: "LP",
        3.0: "TR",
        4.0: "AV",
        5.0: "IC",
    },
    t_bg=50,
    t_cl=25,
):
    _ = t_bg, t_cl
    max_indices = np.argmax(out_np, axis=0)
    processed_out = np.eye(len(out_np))[max_indices].T
    bg_diff = np.diff(processed_out[0])
    if np.abs(bg_diff).sum() != 0:
        start_, end_, _ = longest_event(bg_diff)
    else:
        start_, end_ = 0, len(processed_out[0]) - 1
    predicted_class = processed_out[1:, start_:end_].sum(axis=1).argmax() + 1
    pred_label = clases_ovdas[predicted_class]
    return predicted_class, pred_label, start_, end_


def fill_short_sequences(arr, t_bg=50, t_cl=25):
    max_indices = np.argmax(arr, axis=0)
    processed_out = np.eye(len(arr))[max_indices].T
    for idx in range(1, len(processed_out)):
        arr_clase = processed_out[idx]
        diff = np.diff(arr_clase)
        start_indices = np.where(diff == -1)[0] + 1
        end_indices = np.where(diff == 1)[0] + 1
        if arr_clase[0] == 0:
            start_indices = np.insert(start_indices, 0, 0)
        if arr_clase[-1] == 0:
            end_indices = np.append(end_indices, len(arr_clase))
        for start, end in zip(start_indices, end_indices):
            if end - start < t_cl:
                arr_clase[start:end] = 1
        clase_true = np.where(arr_clase == 1)[0]
        if clase_true.shape[0] != 0:
            processed_out[idx] = arr_clase
            idx_list = [n for n in range(len(processed_out))]
            idx_list.remove(idx)
            processed_out[np.ix_(idx_list, clase_true)] = 0
    arr_bg = processed_out[0]
    diff = np.diff(arr_bg)
    start_indices = np.where(diff == -1)[0] + 1
    end_indices = np.where(diff == 1)[0] + 1
    if arr_bg[0] == 0:
        start_indices = np.insert(start_indices, 0, 0)
    if arr_bg[-1] == 0:
        end_indices = np.append(end_indices, len(arr_bg))
    for start, end in zip(start_indices, end_indices):
        if end - start < t_bg:
            arr_bg[start:end] = 1
    bg_true = np.where(arr_bg == 1)
    processed_out[0] = arr_bg
    processed_out[1:, bg_true] = 0
    return processed_out


def cm_eval(
    model,
    dataloader,
    device,
    len_window=8192,
    im_size=256,
    clases_list={
        1.0: "VT",
        2.0: "LP",
        3.0: "TR",
        4.0: "AV",
        5.0: "IC",
    },
    t_bg=0,
    t_cl=0,
):
    pred_label = []
    true_label = []
    n_classes = len(clases_list) + 1
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for data in dataloader:
            xb, target, _ = data
            xb = xb.to(device)

            target_trace = data_utils.activation_unstacking(
                target,
                len_window=len_window,
                N=im_size,
                n_classes=n_classes,
            )
            true_label_temp = (
                target_trace[:, 1:, :].sum(axis=2).max(axis=1).indices.numpy() + 1
            )
            true_label.extend(true_label_temp.tolist())

            output = model(xb)
            if output.ndim == 4:
                output = torch.softmax(output, dim=1)
            output = output.detach().cpu()
            output_trace = data_utils.activation_unstacking(
                output,
                len_window=len_window,
                N=im_size,
                n_classes=n_classes,
            )
            for idx in range(len(output_trace)):
                out_np = output_trace[idx].numpy()
                pred, _, _, _ = predicted_from_output(
                    out_np,
                    clases_list,
                    t_bg=t_bg,
                    t_cl=t_cl,
                )
                pred_label.append(pred)

            del xb, target, target_trace, output, output_trace

    if was_training:
        model.train()

    cm = confusion_matrix(true_label, pred_label, labels=[1, 2, 3, 4, 5])
    return cm


def random_time_shift(x: np.ndarray, y: np.ndarray, max_shift: int):
    shift = np.random.randint(-max_shift, max_shift + 1)
    x_shifted = np.roll(x, shift=shift, axis=1)
    y_shifted = np.roll(y, shift=shift, axis=1)
    return x_shifted, y_shifted, shift


def amplitude_scaling(x: np.ndarray, low: float = 0.8, high: float = 1.2):
    scale = np.random.uniform(low, high)
    return x * scale, scale


def add_noise(x: np.ndarray, std_factor: float = 0.02):
    x_std = float(np.std(x))
    noise_std = max(std_factor * x_std, 1e-6)
    noise = np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return x + noise, noise_std


def augment_trace(
    x: np.ndarray,
    y: np.ndarray,
    max_shift_samples: int,
    amp_scale_min: float,
    amp_scale_max: float,
    noise_std_factor: float,
):
    # 100% time shift (labels must shift too).
    x_aug, y_aug, shift = random_time_shift(x, y, max_shift_samples)

    did_amp = False
    amp_scale = 1.0
    if np.random.rand() < 0.8:
        x_aug, amp_scale = amplitude_scaling(x_aug, amp_scale_min, amp_scale_max)
        did_amp = True

    did_noise = False
    noise_std = 0.0
    if np.random.rand() < 0.5:
        x_aug, noise_std = add_noise(x_aug, noise_std_factor)
        did_noise = True

    meta = {
        "shift": int(shift),
        "did_amp": did_amp,
        "amp_scale": float(amp_scale),
        "did_noise": did_noise,
        "noise_std": float(noise_std),
    }
    return x_aug.astype(np.float32), y_aug.astype(np.float32), meta


def save_augmentation_plot(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    x_aug: np.ndarray,
    y_aug: np.ndarray,
    out_path: Path,
    title: str,
):
    t = np.arange(x_raw.shape[1])
    raw_labels = np.argmax(y_raw, axis=0)
    aug_labels = np.argmax(y_aug, axis=0)

    # 8 station panels + 1 label panel, with raw/aug interleaved in each station axis.
    n_stations = min(8, x_raw.shape[0], x_aug.shape[0])
    fig, axes = plt.subplots(n_stations + 1, 1, figsize=(12, 14), sharex=True)

    for i in range(n_stations):
        axes[i].plot(t, x_raw[i], lw=0.7, color="black", alpha=0.8, label="raw")
        axes[i].plot(t, x_aug[i], lw=0.7, color="tab:blue", alpha=0.8, label="aug")
        axes[i].set_ylabel(f"S{i+1}")
        if i == 0:
            axes[i].legend(loc="upper right", ncol=2, fontsize=8)

    axes[-1].plot(t, raw_labels, lw=0.9, color="tab:orange", label="raw label")
    axes[-1].plot(
        t, aug_labels, lw=0.9, color="tab:green", alpha=0.8, label="aug label"
    )
    axes[-1].set_ylabel("class")
    axes[-1].set_xlabel("sample")
    axes[-1].legend(loc="upper right", ncol=2, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_validation_event_plot(
    x_raw: np.ndarray,
    out_np: np.ndarray,
    processed_out: np.ndarray,
    out_path: Path,
    title: str,
):
    """Save validation debug plot with traces + raw activations + postprocessed activations."""
    class_names = ["BG", "VT", "LP", "TR", "AV", "IC"]
    class_colors = {
        "BG": "#808080",
        "VT": "#df8d5e",
        "LP": "#2ca02c",
        "TR": "#d62728",
        "AV": "#9467bd",
        "IC": "#8c564b",
    }

    t = np.arange(x_raw.shape[1])
    n_stations = min(8, x_raw.shape[0])
    fig, axes = plt.subplots(
        n_stations + 2,
        1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"hspace": 0.0},
    )

    # 8 station traces, glued vertically.
    for i in range(n_stations):
        ax = axes[i]
        ax.plot(t, x_raw[i], lw=0.7, color="black")
        ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel(f"S{i+1}", rotation=0, labelpad=12, fontsize=8)
        ax.set_xticks([])
        ax.margins(x=0)

    # Raw model activations.
    ax_raw = axes[n_stations]
    for c, cname in enumerate(class_names):
        ax_raw.plot(t, out_np[c], lw=1.0, color=class_colors[cname], label=cname)
    ax_raw.set_ylim(-0.2, 1.2)
    ax_raw.set_ylabel("raw", rotation=0, labelpad=18, fontsize=9)
    ax_raw.legend(loc="upper right", ncol=6, fontsize=8, frameon=False)
    ax_raw.margins(x=0)

    # Postprocessed one-hot activations used for F1 logic.
    ax_proc = axes[n_stations + 1]
    for c, cname in enumerate(class_names):
        ax_proc.plot(
            t,
            processed_out[c],
            lw=1.0,
            color=class_colors[cname],
            label=cname,
        )
    ax_proc.set_ylim(-0.2, 1.2)
    ax_proc.set_ylabel("proc", rotation=0, labelpad=18, fontsize=9)
    ax_proc.set_xlabel("sample")
    ax_proc.margins(x=0)

    for ax in axes[n_stations:]:
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_event_plot_payloads(
    event_plot_payloads: Sequence[dict],
    event_plots_dir: Path,
    epoch: Optional[int] = None,
) -> int:
    """Persist deferred event-plot payloads under an epoch subfolder."""
    epoch_tag = f"epoch_{int(epoch):03d}" if epoch is not None else "epoch_na"
    epoch_plot_dir = event_plots_dir / epoch_tag
    epoch_plot_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for payload in event_plot_payloads:
        out_path = epoch_plot_dir / (
            f"sample_{int(payload['sample_global_idx']):05d}_"
            f"true_{payload['true_name']}_pred_{payload['pred_name']}.png"
        )
        title = (
            f"{epoch_tag} | sample={int(payload['sample_global_idx'])} | "
            f"true={payload['true_name']}({int(payload['true_evt'])}) | "
            f"pred={payload['pred_name']}({int(payload['pred_evt'])})"
        )
        save_validation_event_plot(
            x_raw=payload["x_raw"],
            out_np=payload["out_np"],
            processed_out=payload["processed_out"],
            out_path=out_path,
            title=title,
        )
        saved_count += 1

    return saved_count


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels: np.ndarray, batch_size: int, drop_last: bool = True):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.classes = sorted(np.unique(self.labels).tolist())
        self.class_indices = {
            c: np.where(self.labels == c)[0].astype(np.int64) for c in self.classes
        }

        self.base_per_class = batch_size // len(self.classes)
        self.remainder = batch_size % len(self.classes)

        if drop_last:
            self.n_batches = len(self.labels) // batch_size
        else:
            self.n_batches = int(np.ceil(len(self.labels) / batch_size))

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        class_pools = {}
        class_ptr = {}
        for c, idx in self.class_indices.items():
            pool = idx.copy()
            np.random.shuffle(pool)
            class_pools[c] = pool
            class_ptr[c] = 0

        for _ in range(self.n_batches):
            batch = []
            class_order = np.random.permutation(self.classes)

            for i, c in enumerate(class_order):
                take = self.base_per_class + (1 if i < self.remainder else 0)
                if take == 0:
                    continue

                pool = class_pools[c]
                ptr = class_ptr[c]

                if ptr + take > len(pool):
                    np.random.shuffle(pool)
                    ptr = 0

                batch.extend(pool[ptr : ptr + take].tolist())
                class_ptr[c] = ptr + take
                class_pools[c] = pool

            np.random.shuffle(batch)
            yield batch


def compute_event_f1_iou(model, loader, device):
    """
    Compute class-level F1 and IoU for event classes only (VT/LP/TR/AV/IC -> 1..5).
    Returns:
        f1_per_class: list[float] len=5
        mean_f1: float
        iou_per_class: list[float] len=5
        mean_iou: float
    """
    cm = cm_eval(model, loader, device)
    f1_scores, _, _ = f1_score_from_confusion_matrix(cm)
    support = np.sum(cm, axis=1)
    active_mask = support > 0
    mean_f1 = (
        float(np.mean([f1_scores[i] for i, active in enumerate(active_mask) if active]))
        if np.any(active_mask)
        else 0.0
    )

    iou_per_class = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        denom = tp + fp + fn
        iou_per_class.append(float(tp / denom) if denom > 0 else 0.0)
    mean_iou = (
        float(
            np.mean(
                [iou_per_class[i] for i, active in enumerate(active_mask) if active]
            )
        )
        if np.any(active_mask)
        else 0.0
    )

    return list(f1_scores), mean_f1, iou_per_class, mean_iou


def compute_event_f1_iou_graphsage(
    model,
    loader,
    device,
    descriptor_names: Optional[list[str]] = None,
    return_cm: bool = False,
    return_val_loss: bool = False,
    return_event_plot_payloads: bool = False,
    save_event_plots: bool = False,
    event_plots_dir: Path = None,
    max_event_plots: int = 30,
    epoch: int = None,
):
    """
    Compute class-level F1 and IoU for GraphSAGE model output.

    Supports:
    - [B, C, T] legacy aggregated output
    - [B, S, C, T] per-station output (stations are averaged only for metric logic)

    Returns:
        f1_per_class: list[float] len=5
        mean_f1: float
        iou_per_class: list[float] len=5
        mean_iou: float
        if return_val_loss=True, returns mean_val_loss before cm
        if return_event_plot_payloads=True, returns event_plot_payloads before cm
        if return_cm=True, returns cm as last element
    """
    event_classes = (1, 2, 3, 4, 5)
    class_map = {1.0: "VT", 2.0: "LP", 3.0: "TR", 4.0: "AV", 5.0: "IC"}

    model.eval()
    pred_label = []
    true_label = []
    iou_scores = []
    iou_by_true_class = {c: [] for c in event_classes}
    temporal_intersections = np.zeros(6, dtype=np.int64)
    temporal_unions = np.zeros(6, dtype=np.int64)
    val_loss_sum = 0.0
    val_batch_count = 0
    class_names = ["BG", "VT", "LP", "TR", "AV", "IC"]
    saved_plot_count = 0
    event_plot_payloads = []
    sample_global_idx = 0

    if save_event_plots and event_plots_dir is None:
        event_plots_dir = Path("validation_event_plots")

    def _extract_envelope_tensor(
        descriptor_payload, xb_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Extract envelope descriptor and map it to [B, S, T] on xb device/dtype."""
        if descriptor_payload is None or "envelope" not in descriptor_payload:
            raise ValueError(
                "Model requires envelope (use_envelope=True), but loader batch does not provide descriptor payload with key 'envelope'."
            )

        envelope = descriptor_payload["envelope"]
        if not torch.is_tensor(envelope):
            envelope = torch.as_tensor(envelope)

        envelope = envelope.to(device=xb_tensor.device, dtype=xb_tensor.dtype)

        # Accept [B,S,T], [B,1,S,T], or [B,S,1,T].
        if envelope.ndim == 3:
            pass
        elif envelope.ndim == 4 and envelope.shape[1] == 1:
            envelope = envelope[:, 0, :, :]
        elif envelope.ndim == 4 and envelope.shape[2] == 1:
            envelope = envelope[:, :, 0, :]
        else:
            raise ValueError(
                f"Unsupported envelope shape {tuple(envelope.shape)}; expected [B,S,T] or single-channel 4D variants."
            )

        if envelope.shape != xb_tensor.shape:
            raise ValueError(
                f"Envelope shape {tuple(envelope.shape)} must match waveform shape {tuple(xb_tensor.shape)}."
            )
        return envelope

    def _extract_descriptor_tensor(
        descriptor_payload,
        names: list[str],
        xb_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Build descriptor tensor [B, D, S, Tdesc] in a fixed descriptor order."""
        if descriptor_payload is None:
            raise ValueError("Descriptor payload is missing from loader batch.")

        desc_list = []
        for name in names:
            if name not in descriptor_payload:
                raise ValueError(f"Descriptor '{name}' not found in batch payload.")

            desc = descriptor_payload[name]
            if not torch.is_tensor(desc):
                desc = torch.as_tensor(desc)
            desc = desc.to(device=xb_tensor.device, dtype=xb_tensor.dtype)

            # Accept [B,S,Tdesc], [B,1,S,Tdesc], or [B,S,1,Tdesc].
            if desc.ndim == 3:
                desc_bst = desc
            elif desc.ndim == 4 and desc.shape[1] == 1:
                desc_bst = desc[:, 0, :, :]
            elif desc.ndim == 4 and desc.shape[2] == 1:
                desc_bst = desc[:, :, 0, :]
            else:
                raise ValueError(
                    f"Unsupported descriptor shape for '{name}': {tuple(desc.shape)}."
                )

            if (
                desc_bst.shape[0] != xb_tensor.shape[0]
                or desc_bst.shape[1] != xb_tensor.shape[1]
            ):
                raise ValueError(
                    f"Descriptor '{name}' shape {tuple(desc_bst.shape)} incompatible with waveform shape {tuple(xb_tensor.shape)}."
                )

            desc_list.append(desc_bst)

        return torch.stack(desc_list, dim=1)

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 3:
                raise ValueError(
                    "Loader must return at least (x, y_onehot, y_label) for GraphSAGE metrics."
                )

            xb, y_onehot, y_label = batch[0], batch[1], batch[2]
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)
            y_label = y_label.to(device).long()

            descriptor_payload = None
            volcano_idx_b = None
            if len(batch) > 3:
                extra_1 = batch[3]
                if isinstance(extra_1, dict):
                    descriptor_payload = extra_1
                elif torch.is_tensor(extra_1) and extra_1.ndim <= 1:
                    volcano_idx_b = extra_1
                else:
                    descriptor_payload = extra_1

            if len(batch) > 4:
                extra_2 = batch[4]
                if torch.is_tensor(extra_2) and extra_2.ndim <= 1:
                    volcano_idx_b = extra_2

            if getattr(model, "use_envelope", False):
                envelope_b = _extract_envelope_tensor(descriptor_payload, xb)
            else:
                envelope_b = None

            model_num_desc = int(getattr(model, "num_descriptors", 0))
            if model_num_desc > 0:
                if descriptor_names is None:
                    raise ValueError(
                        "descriptor_names must be provided when model.num_descriptors > 0"
                    )
                if len(descriptor_names) != model_num_desc:
                    raise ValueError(
                        f"descriptor_names length ({len(descriptor_names)}) must match model.num_descriptors ({model_num_desc})."
                    )
                descriptors_b = _extract_descriptor_tensor(
                    descriptor_payload,
                    descriptor_names,
                    xb,
                )
            else:
                descriptors_b = None

            # Edge dynamic features for edge_mpnn__xcorr ablation.
            edge_attr_dynamic_b = None
            if (
                descriptor_payload is not None
                and "edge_attr_dynamic" in descriptor_payload
            ):
                ead = descriptor_payload["edge_attr_dynamic"]
                if not torch.is_tensor(ead):
                    ead = torch.as_tensor(ead)
                edge_attr_dynamic_b = ead.to(device=xb.device, dtype=xb.dtype)

            forward_kwargs = {}
            if envelope_b is not None:
                forward_kwargs["envelope"] = envelope_b
            if descriptors_b is not None:
                forward_kwargs["descriptors"] = descriptors_b
            if edge_attr_dynamic_b is not None:
                forward_kwargs["edge_attr_dynamic"] = edge_attr_dynamic_b
            if volcano_idx_b is not None:
                forward_kwargs["volcano_idx"] = volcano_idx_b.to(device).long()

            output = model(xb, **forward_kwargs)

            if return_val_loss:
                loss, _, _ = combined_dice_ce_loss(
                    output,
                    y_onehot,
                    class_weights=None,
                )
                val_loss_sum += float(loss.item())
                val_batch_count += 1

            if output.ndim == 4:
                # New format [B, S, C, T]: aggregate stations to preserve old metric logic.
                probs = torch.softmax(output, dim=2).mean(dim=1)
            elif output.ndim == 3:
                probs = torch.softmax(output, dim=1)
            else:
                raise ValueError(
                    f"Unexpected GraphSAGE output shape {tuple(output.shape)}; expected [B,C,T] or [B,S,C,T]."
                )

            # Window-level event class for F1/confusion:
            # saturate [C,T] -> one-hot over classes per time, then class-sum over event window.
            pred_evt_list = []
            for b in range(probs.shape[0]):
                out_np = probs[b].detach().cpu().numpy()  # [C, T]
                pred_evt, _, _, _ = predicted_from_output(out_np, class_map)
                pred_evt_list.append(pred_evt)
                true_evt = int(y_label[b].detach().cpu().item())
                is_misclassified = int(pred_evt) != true_evt

                if (
                    (save_event_plots or return_event_plot_payloads)
                    and is_misclassified
                    and saved_plot_count < max_event_plots
                ):
                    max_indices = np.argmax(out_np, axis=0)
                    processed_out = np.eye(len(out_np), dtype=np.float32)[max_indices].T

                    x_raw = xb[b].detach().cpu().numpy()  # [S, T]
                    pred_name = class_names[int(pred_evt)]
                    true_name = class_names[true_evt]

                    if save_event_plots:
                        epoch_tag = (
                            f"epoch_{int(epoch):03d}"
                            if epoch is not None
                            else "epoch_na"
                        )
                        epoch_plot_dir = event_plots_dir / epoch_tag
                        epoch_plot_dir.mkdir(parents=True, exist_ok=True)
                        out_path = epoch_plot_dir / (
                            f"sample_{sample_global_idx:05d}_"
                            f"true_{true_name}_pred_{pred_name}.png"
                        )
                        title = (
                            f"{epoch_tag} | sample={sample_global_idx} | "
                            f"true={true_name}({true_evt}) | pred={pred_name}({int(pred_evt)})"
                        )
                        save_validation_event_plot(
                            x_raw=x_raw,
                            out_np=out_np,
                            processed_out=processed_out,
                            out_path=out_path,
                            title=title,
                        )

                    if return_event_plot_payloads:
                        event_plot_payloads.append(
                            {
                                "sample_global_idx": int(sample_global_idx),
                                "true_evt": int(true_evt),
                                "pred_evt": int(pred_evt),
                                "true_name": true_name,
                                "pred_name": pred_name,
                                "x_raw": x_raw,
                                "out_np": out_np,
                                "processed_out": processed_out,
                            }
                        )
                    saved_plot_count += 1
                sample_global_idx += 1
            pred_evt_batch = torch.as_tensor(
                pred_evt_list, device=device, dtype=torch.long
            )

            # Ground-truth event class from labels (unchanged).
            true_evt_batch = torch.argmax(y_onehot[:, 1:, :].sum(dim=2), dim=1) + 1

            assert torch.equal(
                true_evt_batch, y_label
            ), "Label mismatch: class from y_onehot does not match dataset label_ids"

            pred_evt_np = pred_evt_batch.detach().cpu().numpy()
            true_evt_np = true_evt_batch.detach().cpu().numpy()
            pred_label.extend(pred_evt_np.tolist())
            true_label.extend(true_evt_np.tolist())

            # Legacy IoU-like score (Dice on binary event masks) per sample.
            # Apply the exact same processing to true and predicted outputs:
            # argmax over classes -> BG mask -> event mask.
            true_max_idx = torch.argmax(y_onehot, dim=1)  # [B, T]
            pred_max_idx = torch.argmax(probs, dim=1)  # [B, T]
            true_max_idx_np = true_max_idx.detach().cpu().numpy().reshape(-1)
            pred_max_idx_np = pred_max_idx.detach().cpu().numpy().reshape(-1)
            for c in range(6):
                pred_mask = pred_max_idx_np == c
                true_mask = true_max_idx_np == c
                temporal_intersections[c] += int(
                    np.logical_and(pred_mask, true_mask).sum()
                )
                temporal_unions[c] += int(np.logical_or(pred_mask, true_mask).sum())
            true_bg_mask = (true_max_idx == 0).float()
            pred_bg_mask = (pred_max_idx == 0).float()
            true_event_mask = 1.0 - true_bg_mask
            pred_event_mask = 1.0 - pred_bg_mask

            inter = (pred_event_mask * true_event_mask).sum(dim=1)
            denom = pred_event_mask.sum(dim=1) + true_event_mask.sum(dim=1)
            iou_like_batch = torch.where(
                denom > 0, (2.0 * inter) / denom, torch.zeros_like(denom)
            )

            iou_like_np = iou_like_batch.detach().cpu().numpy()
            iou_scores.extend(iou_like_np.tolist())

            for c in event_classes:
                class_vals = iou_like_np[true_evt_np == c]
                if class_vals.size > 0:
                    iou_by_true_class[c].extend(class_vals.tolist())

            del (
                xb,
                y_onehot,
                y_label,
                output,
                probs,
                pred_evt_batch,
                true_evt_batch,
                true_max_idx,
                pred_max_idx,
                true_bg_mask,
                pred_bg_mask,
                true_event_mask,
                pred_event_mask,
                inter,
                denom,
                iou_like_batch,
            )
            if descriptor_payload is not None:
                del descriptor_payload
            if envelope_b is not None:
                del envelope_b
            if descriptors_b is not None:
                del descriptors_b

    cm = confusion_matrix(true_label, pred_label, labels=list(event_classes))
    f1_scores, _, _ = f1_score_from_confusion_matrix(cm)
    support = np.sum(cm, axis=1)
    active_mask = support > 0
    mean_f1 = (
        float(np.mean([f1_scores[i] for i, active in enumerate(active_mask) if active]))
        if np.any(active_mask)
        else 0.0
    )

    iou_per_class = [
        float(np.mean(iou_by_true_class[c])) if len(iou_by_true_class[c]) > 0 else 0.0
        for c in event_classes
    ]
    mean_iou = (
        float(
            np.mean(
                [iou_per_class[i] for i, active in enumerate(active_mask) if active]
            )
        )
        if np.any(active_mask)
        else 0.0
    )

    # Extra multiclass temporal IoU over all classes [BG, VT, LP, TR, AV, IC].
    iou_all_classes = []
    for c in range(6):
        inter = int(temporal_intersections[c])
        union = int(temporal_unions[c])
        iou_all_classes.append(float(inter / union) if union > 0 else 0.0)
    mean_iou_all = float(np.mean(iou_all_classes))
    mean_val_loss = (
        float(val_loss_sum / val_batch_count) if val_batch_count > 0 else 0.0
    )

    result = (
        list(f1_scores),
        mean_f1,
        iou_per_class,
        mean_iou,
        iou_all_classes,
        mean_iou_all,
    )
    if return_val_loss:
        result = (*result, mean_val_loss)
    if return_event_plot_payloads:
        result = (*result, event_plot_payloads)
    if return_cm:
        return (*result, cm)
    return result


class UNetPatchDataset(Dataset):
    def __init__(
        self,
        npz_path: Path,
        return_debug: bool = False,
        return_meta: bool = False,
    ):
        with np.load(npz_path) as data:
            self.filepaths = data["filepaths"].copy()
            self.labels = data["labels"].copy()
            self.label_ids = data["label_ids"].astype(np.int64, copy=True)
        self.return_debug = return_debug
        self.return_meta = return_meta

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        arr = np.load(self.filepaths[idx], mmap_mode="r")
        arr = np.array(arr, dtype=np.float32)
        x_raw = arr[:8, :]
        y_raw = arr[8:, :]

        x_used, y_used = x_raw, y_raw
        aug_meta = {
            "shift": 0,
            "did_amp": False,
            "amp_scale": 1.0,
            "did_noise": False,
            "noise_std": 0.0,
        }

        x = torch.from_numpy(x_used).unsqueeze(0)
        y = torch.from_numpy(y_used)

        x_unet = data_utils.patch_stacking_X(x, N=256).squeeze(0)
        y_onehot = data_utils.patch_stacking_y(
            y,
            N=256,
            n_classes=6,
            n_stations=8,
        ).squeeze(0)
        y_idx = torch.argmax(y_onehot, dim=0).long()

        if self.return_debug:
            return x_unet, y_onehot, y_idx, x_raw, y_raw, x_used, y_used, aug_meta

        if self.return_meta:
            return (
                x_unet,
                y_onehot,
                y_idx,
                self.labels[idx],
                Path(self.filepaths[idx]).name,
            )

        return x_unet, y_onehot, y_idx


class GraphSAGEDataset(Dataset):
    """
    Dataset for multi-station seismic segmentation with GraphSAGE.

    Returns data in [B, S, T] format without patch stacking:
    - x: [S, T] multi-station waveforms (S=8 stations, T=8192 samples)
    - y: [6, S, T] one-hot encoded labels (6 classes)
    - y_idx: [S, T] class indices

    Optional descriptors can be returned as stacked tensor [D, S, T], where D is
    the number of selected descriptors.
    """

    AVAILABLE_DESCRIPTORS = (
        "envelope",
        "dominant_frequency",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_entropy",
    )

    def __init__(
        self,
        npz_path: Path,
        descriptor_names: Sequence[str] | str | None = None,
        edge_data_npz: Optional[Path] = None,
        return_volcano_idx: bool = False,
        volcano_name_to_idx: Optional[dict[str, int]] = None,
    ):
        with np.load(npz_path) as data:
            self.filepaths = data["filepaths"].copy()
            self.descriptor_paths = (
                data["descriptor_paths"].copy() if "descriptor_paths" in data else None
            )
            self.labels = data["labels"].copy()
            self.label_ids = data["label_ids"].astype(np.int64, copy=True)

        self.descriptor_names = self._normalize_descriptor_names(descriptor_names)
        self.use_descriptors = len(self.descriptor_names) > 0
        self.return_volcano_idx = bool(return_volcano_idx)
        self.volcano_name_to_idx = (
            dict(volcano_name_to_idx)
            if volcano_name_to_idx is not None
            else get_default_volcano_to_index()
        )

        if self.return_volcano_idx:
            self.sample_volcano_idx = np.asarray(
                [
                    int(self.volcano_name_to_idx[infer_volcano_name_from_path(str(fp))])
                    for fp in self.filepaths
                ],
                dtype=np.int64,
            )
        else:
            self.sample_volcano_idx = None

        if self.use_descriptors and self.descriptor_paths is None:
            self.descriptor_paths = self._infer_descriptor_paths(npz_path)

        # Precomputed cross-correlation edge features for edge_mpnn__xcorr.
        self._edge_attr_dynamic: Optional[np.ndarray] = None
        if edge_data_npz is not None:
            edge_data_npz = Path(edge_data_npz)
            if not edge_data_npz.exists():
                raise FileNotFoundError(
                    f"Edge data file not found: {edge_data_npz}. "
                    "Run scripts/01b_edge_data.py first."
                )
            with np.load(edge_data_npz) as ed:
                self._edge_attr_dynamic = ed["edge_attr_dynamic"].astype(
                    np.float32, copy=True
                )
            if len(self._edge_attr_dynamic) != len(self.filepaths):
                raise ValueError(
                    f"Edge data length {len(self._edge_attr_dynamic)} does not match "
                    f"manifest length {len(self.filepaths)} for {edge_data_npz}."
                )

    @classmethod
    def _normalize_descriptor_names(
        cls,
        descriptor_names: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        if descriptor_names is None:
            return tuple()
        if isinstance(descriptor_names, str):
            if descriptor_names.lower() == "all":
                return tuple(cls.AVAILABLE_DESCRIPTORS)
            names = (descriptor_names,)
        else:
            names = tuple(descriptor_names)

        unknown = sorted(set(names) - set(cls.AVAILABLE_DESCRIPTORS))
        if len(unknown) > 0:
            raise ValueError(
                "Unknown descriptor names: "
                f"{unknown}. Available: {list(cls.AVAILABLE_DESCRIPTORS)}"
            )
        return tuple(names)

    def _infer_descriptor_paths(self, npz_path: Path) -> np.ndarray:
        """
        Backward-compatible path inference for manifests without descriptor_paths.

        Expected descriptor layout:
            prepared/descriptors/<VOLCANO>/<class>/<event>.npz
        """
        npz_path = Path(npz_path)
        volcano_name = npz_path.parent.name
        descriptors_root = npz_path.parent.parent / "descriptors" / volcano_name

        inferred = []
        for fp in self.filepaths:
            src = Path(str(fp))
            if volcano_name in src.parts:
                idx = src.parts.index(volcano_name)
                rel = Path(*src.parts[idx + 1 :])
            else:
                rel = src.name
            inferred.append(
                str((descriptors_root / rel).with_suffix(".npz").as_posix())
            )
        return np.array(inferred)

    def _load_selected_descriptors(self, idx: int) -> dict[str, torch.Tensor]:
        desc_path = Path(str(self.descriptor_paths[idx]))
        if not desc_path.exists():
            raise FileNotFoundError(
                f"Descriptor file not found for sample index {idx}: {desc_path}"
            )

        with np.load(desc_path, mmap_mode="r") as desc_npz:
            desc_dict = {
                name: torch.from_numpy(np.array(desc_npz[name], dtype=np.float32))
                for name in self.descriptor_names
            }
        return desc_dict

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        arr = np.load(self.filepaths[idx], mmap_mode="r")
        arr = np.array(arr, dtype=np.float32)
        x_raw = arr[:8, :]  # [S=8, T=8192]
        y_raw = arr[8:, :]  # [C=6, T=8192]

        x_used, y_used = x_raw, y_raw
        aug_meta = {
            "shift": 0,
            "did_amp": False,
            "amp_scale": 1.0,
            "did_noise": False,
            "noise_std": 0.0,
        }

        # Convert to tensors without patch stacking
        x = torch.from_numpy(x_used)  # [S=8, T=8192]
        y = torch.from_numpy(y_used)  # [C=6, T=8192]

        # Labels are same across all stations, return as [C=6, T=8192]
        y_onehot = y  # [C=6, T=8192]

        # Get class indices: [C=6, T=8192] -> [T=8192]
        y_idx = torch.argmax(y_onehot, dim=0).long()  # [T]
        y_label = torch.tensor(int(self.label_ids[idx]), dtype=torch.long)
        descriptors = (
            self._load_selected_descriptors(idx) if self.use_descriptors else None
        )
        volcano_idx = (
            torch.tensor(int(self.sample_volcano_idx[idx]), dtype=torch.long)
            if self.return_volcano_idx
            else None
        )

        if self.use_descriptors and self.return_volcano_idx:
            return x, y_onehot, y_label, descriptors, volcano_idx

        if self.use_descriptors or self._edge_attr_dynamic is not None:
            return x, y_onehot, y_label, descriptors

        if self.return_volcano_idx:
            return x, y_onehot, y_label, volcano_idx

        return x, y_onehot, y_label


class CrossVolcanoLOODataset(Dataset):
    """
    Dataset for leave-one-out cross-volcano protocol manifests.

    Expected manifest fields:
    - filepaths
    - labels
    - label_ids
    Optional:
    - volcano_idx
    - descriptor_paths
    """

    AVAILABLE_DESCRIPTORS = GraphSAGEDataset.AVAILABLE_DESCRIPTORS

    def __init__(
        self,
        npz_path: Path,
        descriptor_names: Sequence[str] | str | None = None,
        edge_data_npz: Optional[Path] = None,
        return_volcano_idx: bool = True,
        volcano_name_to_idx: Optional[dict[str, int]] = None,
    ):
        with np.load(npz_path) as data:
            self.filepaths = data["filepaths"].copy()
            self.labels = data["labels"].copy()
            self.label_ids = data["label_ids"].astype(np.int64, copy=True)
            self.descriptor_paths = (
                data["descriptor_paths"].copy() if "descriptor_paths" in data else None
            )
            self.manifest_volcano_idx = (
                data["volcano_idx"].astype(np.int64, copy=True)
                if "volcano_idx" in data
                else None
            )

        self.descriptor_names = GraphSAGEDataset._normalize_descriptor_names(
            descriptor_names
        )
        self.use_descriptors = len(self.descriptor_names) > 0
        self.return_volcano_idx = bool(return_volcano_idx)
        self.volcano_name_to_idx = (
            dict(volcano_name_to_idx)
            if volcano_name_to_idx is not None
            else get_default_volcano_to_index()
        )

        if self.manifest_volcano_idx is not None:
            self.sample_volcano_idx = self.manifest_volcano_idx
        else:
            self.sample_volcano_idx = np.asarray(
                [
                    int(self.volcano_name_to_idx[infer_volcano_name_from_path(str(fp))])
                    for fp in self.filepaths
                ],
                dtype=np.int64,
            )

        if self.use_descriptors and self.descriptor_paths is None:
            self.descriptor_paths = self._infer_descriptor_paths()

        # Optional precomputed dynamic edge features for xcorr ablations.
        self._edge_attr_dynamic: Optional[np.ndarray] = None
        if edge_data_npz is not None:
            edge_data_npz = Path(edge_data_npz)
            if not edge_data_npz.exists():
                raise FileNotFoundError(
                    f"Edge data file not found: {edge_data_npz}. "
                    "Run scripts/01b_edge_data.py first."
                )
            with np.load(edge_data_npz) as ed:
                self._edge_attr_dynamic = ed["edge_attr_dynamic"].astype(
                    np.float32, copy=True
                )
            if len(self._edge_attr_dynamic) != len(self.filepaths):
                raise ValueError(
                    f"Edge data length {len(self._edge_attr_dynamic)} does not match "
                    f"manifest length {len(self.filepaths)} for {edge_data_npz}."
                )

    def _infer_descriptor_paths(self) -> np.ndarray:
        project_root = Path(__file__).resolve().parents[1]
        descriptors_root = project_root / "data" / "prepared_data" / "descriptors"
        inferred: list[str] = []
        for fp in self.filepaths:
            src = Path(str(fp))
            try:
                volcano_name = infer_volcano_name_from_path(src)
            except KeyError:
                inferred.append("")
                continue

            parts = src.parts
            if volcano_name in parts:
                idx = parts.index(volcano_name)
                rel = Path(*parts[idx + 1 :]).with_suffix(".npz")
            else:
                rel = Path(src.name).with_suffix(".npz")

            desc_path = descriptors_root / volcano_name / rel
            inferred.append(str(desc_path.as_posix()))
        return np.asarray(inferred)

    def _load_selected_descriptors(self, idx: int) -> dict[str, torch.Tensor]:
        desc_path = Path(str(self.descriptor_paths[idx]))
        if not desc_path.exists():
            raise FileNotFoundError(
                f"Descriptor file not found for sample index {idx}: {desc_path}"
            )

        with np.load(desc_path, mmap_mode="r") as desc_npz:
            desc_dict = {
                name: torch.from_numpy(np.array(desc_npz[name], dtype=np.float32))
                for name in self.descriptor_names
            }
        return desc_dict

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        arr = np.load(self.filepaths[idx], mmap_mode="r")
        arr = np.array(arr, dtype=np.float32)
        x_raw = arr[:8, :]
        y_raw = arr[8:, :]

        x = torch.from_numpy(x_raw)
        y_onehot = torch.from_numpy(y_raw)
        y_label = torch.tensor(int(self.label_ids[idx]), dtype=torch.long)
        descriptors: dict = (
            self._load_selected_descriptors(idx) if self.use_descriptors else {}
        )
        if self._edge_attr_dynamic is not None:
            descriptors["edge_attr_dynamic"] = torch.from_numpy(
                self._edge_attr_dynamic[idx].copy()
            )
        volcano_idx = (
            torch.tensor(int(self.sample_volcano_idx[idx]), dtype=torch.long)
            if self.return_volcano_idx
            else None
        )

        has_payload = self.use_descriptors or self._edge_attr_dynamic is not None

        if has_payload and self.return_volcano_idx:
            return x, y_onehot, y_label, descriptors, volcano_idx
        if has_payload:
            return x, y_onehot, y_label, descriptors
        if self.return_volcano_idx:
            return x, y_onehot, y_label, volcano_idx
        return x, y_onehot, y_label


def compute_iou_per_class(
    pred_idx: np.ndarray, true_idx: np.ndarray, n_classes: int = 6
):
    ious = []
    for c in range(n_classes):
        pred_mask = pred_idx == c
        true_mask = true_idx == c
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return np.array(ious)


def f1_score_from_confusion_matrix(confusion_matrix: np.ndarray):
    f1_scores = []
    recall_scores = []
    precision_scores = []
    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)
    return f1_scores, recall_scores, precision_scores


def event_iou_like_score(pred_idx_window: np.ndarray, true_idx_window: np.ndarray):
    pred_event = (pred_idx_window != 0).astype(np.int32)
    true_event = (true_idx_window != 0).astype(np.int32)
    denom = pred_event.sum() + true_event.sum()
    if denom == 0:
        return 1.0
    inter = (pred_event * true_event).sum()
    return float((2.0 * inter) / denom)


def cleanup_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def ensure_fold_data_exists(fold_data_dir: Path) -> None:
    needed = [
        fold_data_dir / "train_aug.npz",
        fold_data_dir / "val.npz",
        fold_data_dir / "test.npz",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing fold manifest files:\n" + "\n".join(missing))


def train_one_ablation_fold(
    ablation_name: str,
    model_kwargs: dict,
    fold_id: int,
    fold_data_dir: Path,
    fold_out_dir: Path,
    device: torch.device,
    config: dict,
) -> dict:

    checkpoints_dir = fold_out_dir / "checkpoints"
    reports_dir = fold_out_dir / "reports"
    cm_dir = fold_out_dir / "confusion_matrices"
    val_plot_dir = fold_out_dir / "validation_event_plots"

    for p in (checkpoints_dir, reports_dir, cm_dir, val_plot_dir):
        p.mkdir(parents=True, exist_ok=True)

    # edge_feature_mode="delta_pos_xcorr" requires precomputed edge data (01b_edge_data.py).
    _needs_xcorr = model_kwargs.get("edge_feature_mode") == "delta_pos_xcorr"

    def _edge_npz(split_name: str) -> Optional[Path]:
        if not _needs_xcorr:
            return None
        return fold_data_dir / "edge_data" / split_name

    train_ds = GraphSAGEDataset(
        fold_data_dir / "train_aug.npz",
        edge_data_npz=_edge_npz("train_aug.npz"),
    )
    val_ds = GraphSAGEDataset(
        fold_data_dir / "val.npz",
        edge_data_npz=_edge_npz("val.npz"),
    )
    test_ds = GraphSAGEDataset(
        fold_data_dir / "test.npz",
        edge_data_npz=_edge_npz("test.npz"),
    )

    balanced_batch_sampler = BalancedBatchSampler(
        train_ds.label_ids, batch_size=config["batch_size"]
    )
    train_loader = DataLoader(train_ds, batch_sampler=balanced_batch_sampler)
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    # Extract the model class if one is provided by the registry.
    model_kwargs_copy = model_kwargs.copy()
    model_class = model_kwargs_copy.pop("_model_cls", None)
    if model_class is None:
        model_class_name = model_kwargs_copy.pop("_model_class", "UNet_GraphSAGE")
        model_class = UNet_MPNN if model_class_name == "UNet_MPNN" else UNet_GraphSAGE
    else:
        model_class_name = getattr(model_class, "__name__", str(model_class))

    model_kwargs_copy.pop("in_channels", None)
    model_kwargs_copy.pop("out_channels", None)

    model = model_class(
        in_channels=1,
        out_channels=6,
        **model_kwargs_copy,
    ).to(device)

    model_name = (
        f"{model_class_name}_{ablation_name}_{config['volcano']}_fold_{fold_id:02d}"
    )

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["epochs"] / 4)),
        eta_min=config["lr_final"],
    )

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_mean_f1 = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    metrics_rows = []
    fold_start = time.time()

    print("=" * 80)
    print(
        f"Training {ablation_name} | fold={fold_id:02d} | "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )
    print(f"Output folder: {fold_out_dir}")
    print("=" * 80)

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            xb = batch[0].to(device)
            y_onehot = batch[1].to(device)
            _train_payload = batch[3] if len(batch) > 3 else None

            _train_edge_attr = None
            if _train_payload is not None and "edge_attr_dynamic" in _train_payload:
                ead = _train_payload["edge_attr_dynamic"]
                if not torch.is_tensor(ead):
                    ead = torch.as_tensor(ead)
                _train_edge_attr = ead.to(device=device, dtype=xb.dtype)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb, edge_attr_dynamic=_train_edge_attr)
            loss, dice_component, ce_component = combined_dice_ce_loss(
                out,
                y_onehot,
                class_weights=None,
                dice_weight=config["dice_weight"],
                ce_weight=config["ce_weight"],
            )
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

            if batch_idx % 100 == 0:
                print(
                    f"  Epoch {epoch:03d} batch {batch_idx:04d}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} dice={dice_component.item():.4f} ce={ce_component.item():.4f}"
                )

            del (
                xb,
                y_onehot,
                out,
                loss,
                dice_component,
                ce_component,
                _train_payload,
                _train_edge_attr,
            )

        scheduler.step()

        (
            f1_per_class,
            mean_f1,
            iou_per_class,
            mean_iou,
            iou_all_classes,
            mean_iou_all,
            val_loss,
            event_plot_payloads,
            cm,
        ) = compute_event_f1_iou_graphsage(
            model,
            val_loader,
            device,
            return_cm=True,
            return_val_loss=True,
            return_event_plot_payloads=True,
            save_event_plots=False,
            event_plots_dir=val_plot_dir,
            max_event_plots=config["val_plot_events"],
            epoch=epoch,
        )

        is_best_mean_f1_epoch = float(mean_f1) > float(best_mean_f1)
        if is_best_mean_f1_epoch:
            saved_plot_count = save_event_plot_payloads(
                event_plot_payloads,
                val_plot_dir,
                epoch=epoch,
            )
            best_mean_f1 = float(mean_f1)
            best_epoch = int(epoch)
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(val_loss),
                    "f1score": float(mean_f1),
                },
                checkpoints_dir / "best_f1.pt",
            )
        else:
            saved_plot_count = 0
            epochs_without_improvement += 1

        if float(train_loss) < float(best_train_loss):
            best_train_loss = float(train_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(val_loss),
                    "f1score": float(mean_f1),
                },
                checkpoints_dir / "best_train_loss.pt",
            )

        if float(val_loss) < float(best_val_loss):
            best_val_loss = float(val_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(val_loss),
                    "f1score": float(mean_f1),
                },
                checkpoints_dir / "best_val_loss.pt",
            )

        if config["save_confusion_matrix_each_epoch"]:
            cm_labels = ["VT", "LP", "TR", "AV", "IC"]
            cm_path = cm_dir / f"confusion_matrix_epoch_{epoch:03d}.png"
            save_confusion_matrix_image(
                cm=cm,
                labels=cm_labels,
                out_path=cm_path,
                title=f"Confusion Matrix - {model_name} - Epoch {epoch}",
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        metrics_rows.append(
            [
                current_lr,
                epoch,
                float(train_loss),
                float(val_loss),
                float(f1_per_class[0]),
                float(f1_per_class[1]),
                float(f1_per_class[2]),
                float(f1_per_class[3]),
                float(f1_per_class[4]),
                float(mean_f1),
                float(iou_per_class[0]),
                float(iou_per_class[1]),
                float(iou_per_class[2]),
                float(iou_per_class[3]),
                float(iou_per_class[4]),
                float(mean_iou),
                float(iou_all_classes[0]),
                float(iou_all_classes[1]),
                float(iou_all_classes[2]),
                float(iou_all_classes[3]),
                float(iou_all_classes[4]),
                float(iou_all_classes[5]),
                float(mean_iou_all),
            ]
        )

        metrics_df = pd.DataFrame(
            metrics_rows,
            columns=[
                "lr",
                "epoch",
                "train_loss",
                "val_loss",
                "VT_f1",
                "LP_f1",
                "TR_f1",
                "AV_f1",
                "IC_f1",
                "mean_f1",
                "VT_iou",
                "LP_iou",
                "TR_iou",
                "AV_iou",
                "IC_iou",
                "mean_iou",
                "BG_iou_all",
                "VT_iou_all",
                "LP_iou_all",
                "TR_iou_all",
                "AV_iou_all",
                "IC_iou_all",
                "mean_iou_all",
            ],
        )
        metrics_df.to_csv(
            reports_dir / "training_metrics.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )

        print(
            f"EPOCH {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"mean_f1={mean_f1:.4f} mean_iou={mean_iou:.4f} "
            f"best_epoch={best_epoch if best_epoch >= 0 else 'NA'} "
            f"no_improve={epochs_without_improvement}/{config['early_stop_patience']} "
            f"saved_best_plots={saved_plot_count}"
        )

        del event_plot_payloads, cm
        cleanup_gpu_cache()

        if epochs_without_improvement >= int(config["early_stop_patience"]):
            print(
                f"Early stopping at epoch {epoch:03d}: no mean_f1 improvement for "
                f"{config['early_stop_patience']} consecutive epochs."
            )
            break

    best_f1_ckpt = checkpoints_dir / "best_f1.pt"
    if not best_f1_ckpt.exists():
        raise RuntimeError(
            f"best_f1 checkpoint not found for fold output: {best_f1_ckpt}"
        )

    ckpt = torch.load(best_f1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    (
        test_f1_per_class,
        test_mean_f1,
        test_iou_per_class,
        test_mean_iou,
        test_iou_all_classes,
        test_mean_iou_all,
        test_loss,
        test_cm,
    ) = compute_event_f1_iou_graphsage(
        model,
        test_loader,
        device,
        return_cm=True,
        return_val_loss=True,
        return_event_plot_payloads=False,
        save_event_plots=False,
        max_event_plots=0,
        epoch=None,
    )

    test_cm_path = cm_dir / "confusion_matrix_test_best_f1.png"
    save_confusion_matrix_image(
        cm=test_cm,
        labels=["VT", "LP", "TR", "AV", "IC"],
        out_path=test_cm_path,
        title=f"Test Confusion Matrix - {model_name} - best_f1",
    )

    fold_elapsed_sec = float(time.time() - fold_start)

    fold_summary = {
        "ablation": ablation_name,
        "fold": int(fold_id),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "best_val_mean_f1": float(best_mean_f1),
        "test_loss": float(test_loss),
        "test_mean_f1": float(test_mean_f1),
        "test_mean_iou": float(test_mean_iou),
        "test_mean_iou_all": float(test_mean_iou_all),
        "test_f1_per_class": [float(x) for x in test_f1_per_class],
        "test_iou_per_class": [float(x) for x in test_iou_per_class],
        "test_iou_all_classes": [float(x) for x in test_iou_all_classes],
        "fold_elapsed_seconds": fold_elapsed_sec,
    }

    with (reports_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    del train_ds, val_ds, test_ds
    del train_loader, val_loader, test_loader
    del optimizer, scheduler, model, ckpt
    cleanup_gpu_cache()

    return fold_summary


def evaluate_unet_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    len_window: int,
    im_size: int,
    config: dict,
) -> tuple[list[float], float, list[float], float, float, np.ndarray]:
    model.eval()

    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for xb, y_onehot, _y_idx in dataloader:
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)
            out = model(xb)
            loss, _, _ = combined_dice_ce_loss_2d(
                out,
                y_onehot,
                class_weights=None,
                dice_weight=config["dice_weight"],
                ce_weight=config["ce_weight"],
            )
            total_loss += float(loss.item())
            n_batches += 1
            del xb, y_onehot, out, loss

    mean_loss = float(total_loss / n_batches) if n_batches > 0 else 0.0

    cm = cm_eval(
        model=model,
        dataloader=dataloader,
        device=device,
        len_window=len_window,
        im_size=im_size,
        clases_list={1.0: "VT", 2.0: "LP", 3.0: "TR", 4.0: "AV", 5.0: "IC"},
        t_bg=0,
        t_cl=0,
    )
    f1_scores, _, _ = f1_score_from_confusion_matrix(cm)
    f1_scores = [float(x) for x in f1_scores]
    mean_f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0
    iou_per_class, mean_iou = compute_iou_from_cm(cm)

    return f1_scores, mean_f1, iou_per_class, mean_iou, mean_loss, cm


def train_one_unet_fold(
    model_key: str,
    fold_id: int,
    fold_data_dir: Path,
    fold_out_dir: Path,
    device: torch.device,
    config: dict,
) -> dict:
    checkpoints_dir = fold_out_dir / "checkpoints"
    reports_dir = fold_out_dir / "reports"
    cm_dir = fold_out_dir / "confusion_matrices"
    for p in (checkpoints_dir, reports_dir, cm_dir):
        p.mkdir(parents=True, exist_ok=True)

    train_ds = UNetPatchDataset(fold_data_dir / "train_aug.npz")
    val_ds = UNetPatchDataset(fold_data_dir / "val.npz")
    test_ds = UNetPatchDataset(fold_data_dir / "test.npz")

    balanced_batch_sampler = BalancedBatchSampler(
        train_ds.label_ids,
        batch_size=config["batch_size"],
    )
    train_loader = DataLoader(train_ds, batch_sampler=balanced_batch_sampler)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    spec = get_model_spec(model_key)
    model = spec["model_cls"](**spec["model_kwargs"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["epochs"] / 4)),
        eta_min=config["lr_final"],
    )

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_mean_f1 = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    metrics_rows = []
    fold_start = time.time()

    print("=" * 80)
    print(
        f"Training {spec['display_name']} | fold={fold_id:02d} | "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )
    print(f"Output folder: {fold_out_dir}")
    print("=" * 80)

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0

        for batch_idx, (xb, y_onehot, _y_idx) in enumerate(train_loader):
            xb = xb.to(device)
            y_onehot = y_onehot.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss, dice_component, ce_component = combined_dice_ce_loss_2d(
                out,
                y_onehot,
                class_weights=None,
                dice_weight=config["dice_weight"],
                ce_weight=config["ce_weight"],
            )
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

            if batch_idx % 100 == 0:
                print(
                    f"  Epoch {epoch:03d} batch {batch_idx:04d}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} dice={dice_component.item():.4f} ce={ce_component.item():.4f}"
                )

            del xb, y_onehot, out, loss, dice_component, ce_component

        scheduler.step()

        (
            val_f1_per_class,
            val_mean_f1,
            val_iou_per_class,
            val_mean_iou,
            val_loss,
            val_cm,
        ) = evaluate_unet_model(
            model=model,
            dataloader=val_loader,
            device=device,
            len_window=config["len_window"],
            im_size=config["im_size"],
            config=config,
        )

        is_best_mean_f1_epoch = float(val_mean_f1) > float(best_mean_f1)
        if is_best_mean_f1_epoch:
            best_mean_f1 = float(val_mean_f1)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(val_loss),
                    "f1score": float(val_mean_f1),
                },
                checkpoints_dir / "best_f1.pt",
            )
        else:
            epochs_without_improvement += 1

        if float(train_loss) < float(best_train_loss):
            best_train_loss = float(train_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(val_loss),
                    "f1score": float(val_mean_f1),
                },
                checkpoints_dir / "best_train_loss.pt",
            )

        if float(val_loss) < float(best_val_loss):
            best_val_loss = float(val_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(val_loss),
                    "f1score": float(val_mean_f1),
                },
                checkpoints_dir / "best_val_loss.pt",
            )

        save_confusion_matrix_image(
            cm=val_cm,
            labels=["VT", "LP", "TR", "AV", "IC"],
            out_path=cm_dir / f"confusion_matrix_epoch_{epoch:03d}.png",
            title=(
                f"Confusion Matrix - {spec['display_name']} "
                f"fold {fold_id:02d} - epoch {epoch:03d}"
            ),
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        metrics_rows.append(
            [
                current_lr,
                int(epoch),
                float(train_loss),
                float(val_loss),
                float(val_f1_per_class[0]),
                float(val_f1_per_class[1]),
                float(val_f1_per_class[2]),
                float(val_f1_per_class[3]),
                float(val_f1_per_class[4]),
                float(val_mean_f1),
                float(val_iou_per_class[0]),
                float(val_iou_per_class[1]),
                float(val_iou_per_class[2]),
                float(val_iou_per_class[3]),
                float(val_iou_per_class[4]),
                float(val_mean_iou),
            ]
        )
        metrics_df = pd.DataFrame(
            metrics_rows,
            columns=[
                "lr",
                "epoch",
                "train_loss",
                "val_loss",
                "VT_f1",
                "LP_f1",
                "TR_f1",
                "AV_f1",
                "IC_f1",
                "mean_f1",
                "VT_iou",
                "LP_iou",
                "TR_iou",
                "AV_iou",
                "IC_iou",
                "mean_iou",
            ],
        )
        metrics_df.to_csv(
            reports_dir / "training_metrics.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
        )

        print(
            f"EPOCH {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"mean_f1={val_mean_f1:.4f} mean_iou={val_mean_iou:.4f} "
            f"best_epoch={best_epoch if best_epoch >= 0 else 'NA'} "
            f"no_improve={epochs_without_improvement}/{config['early_stop_patience']}"
        )

        del val_cm
        cleanup_gpu_cache()

        if epochs_without_improvement >= int(config["early_stop_patience"]):
            print(
                f"Early stopping at epoch {epoch:03d}: no mean_f1 improvement for "
                f"{config['early_stop_patience']} consecutive epochs."
            )
            break

    best_f1_ckpt = checkpoints_dir / "best_f1.pt"
    if not best_f1_ckpt.exists():
        raise RuntimeError(
            f"best_f1 checkpoint not found for fold output: {best_f1_ckpt}"
        )

    ckpt = torch.load(best_f1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    (
        test_f1_per_class,
        test_mean_f1,
        test_iou_per_class,
        test_mean_iou,
        test_loss,
        test_cm,
    ) = evaluate_unet_model(
        model=model,
        dataloader=test_loader,
        device=device,
        len_window=config["len_window"],
        im_size=config["im_size"],
        config=config,
    )

    save_confusion_matrix_image(
        cm=test_cm,
        labels=["VT", "LP", "TR", "AV", "IC"],
        out_path=cm_dir / "confusion_matrix_test_best_f1.png",
        title=(
            f"Test Confusion Matrix - {spec['display_name']} "
            f"fold {fold_id:02d} - best_f1"
        ),
    )

    fold_elapsed_sec = float(time.time() - fold_start)
    fold_summary = {
        "model": spec["display_name"],
        "model_key": model_key,
        "fold": int(fold_id),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "best_val_mean_f1": float(best_mean_f1),
        "test_loss": float(test_loss),
        "test_mean_f1": float(test_mean_f1),
        "test_mean_iou": float(test_mean_iou),
        "test_f1_per_class": [float(x) for x in test_f1_per_class],
        "test_iou_per_class": [float(x) for x in test_iou_per_class],
        "fold_elapsed_seconds": fold_elapsed_sec,
    }

    with (reports_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    del train_ds, val_ds, test_ds
    del train_loader, val_loader, test_loader
    del optimizer, scheduler, model, ckpt
    cleanup_gpu_cache()

    return fold_summary
