import gc
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import BatchSampler, Dataset

from utils.evaluation.metrics_core import (
    event_f1_agnostic_from_confusion_matrix,
    event_iou_active_only_from_class_indices,
    summarize_scalar_values,
)
def dice_loss_multistation(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    class_weights: Optional[torch.Tensor] = None,
):
    """Dice loss for multi-station outputs [B,C,T] or [B,S,C,T]."""
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


def classwise_dice_multistation(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Class-wise Dice scores for multi-station outputs [B,C,T] or [B,S,C,T]."""
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
    dice = classwise_dice_multistation(pred, target_onehot, smooth=1e-6)
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
    processed_out = fill_short_sequences(out_np, t_bg=t_bg, t_cl=t_cl)
    # If no event survives postprocessing, return BG over full window.
    if float(processed_out[1:, :].sum()) <= 0.0:
        return 0, "BG", 0, len(processed_out[0]) - 1

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
    true_out: Optional[np.ndarray] = None,
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
    n_extra_panels = 3 if true_out is not None else 2
    fig, axes = plt.subplots(
        n_stations + n_extra_panels,
        1,
        figsize=(14, 11 + int(true_out is not None)),
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
    if true_out is None:
        ax_proc.set_xlabel("sample")
    ax_proc.margins(x=0)

    if true_out is not None:
        ax_true = axes[n_stations + 2]
        for c, cname in enumerate(class_names):
            ax_true.plot(
                t,
                true_out[c],
                lw=1.0,
                color=class_colors[cname],
                label=cname,
            )
        ax_true.set_ylim(-0.2, 1.2)
        ax_true.set_ylabel("true", rotation=0, labelpad=18, fontsize=9)
        ax_true.set_xlabel("sample")
        ax_true.margins(x=0)

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
            true_out=payload.get("y_onehot"),
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


def compute_event_f1_iou_multistation(
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
    Compute class-level F1 and IoU for 1D multi-station model output.

    Supports:
    - [B, C, T] native graph-model output
    - [B, S, C, T] legacy compatibility path; stations are reduced before metrics

    Returns:
        f1_per_class: list[float] len=6
        mean_f1: float
        mean_iou: class-agnostic event-vs-background IoU over time
        if return_val_loss=True, returns mean_val_loss before cm
        if return_event_plot_payloads=True, returns event_plot_payloads before cm
        if return_cm=True, returns cm as last element
    """
    event_classes = (0, 1, 2, 3, 4, 5)
    class_map = {1.0: "VT", 2.0: "LP", 3.0: "TR", 4.0: "AV", 5.0: "IC"}

    model.eval()
    pred_label = []
    true_label = []
    event_intersection = 0
    event_union = 0
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
                    "Loader must return at least (x, y_onehot, y_label) for multistation metrics."
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
                # Reduce station dimension before temporal postprocessing.
                probs = torch.softmax(output, dim=2).mean(dim=1)
            elif output.ndim == 3:
                probs = torch.softmax(output, dim=1)
            else:
                raise ValueError(
                    f"Unexpected multistation output shape {tuple(output.shape)}; expected [B,C,T] or [B,S,C,T]."
                )

            # Window-level event class for F1/confusion.
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
                                "y_onehot": y_onehot[b].detach().cpu().numpy(),
                            }
                        )
                    saved_plot_count += 1
                sample_global_idx += 1
            pred_evt_batch = torch.as_tensor(
                pred_evt_list, device=device, dtype=torch.long
            )

            # Keep metrics over all classes, including BG=0.
            true_evt_batch = y_label

            pred_evt_np = pred_evt_batch.detach().cpu().numpy()
            true_evt_np = true_evt_batch.detach().cpu().numpy()
            pred_label.extend(pred_evt_np.tolist())
            true_label.extend(true_evt_np.tolist())

            # Class-agnostic event-vs-background IoU over time.
            true_max_idx = torch.argmax(y_onehot, dim=1)  # [B, T]
            pred_max_idx = torch.argmax(probs, dim=1)  # [B, T]
            mean_iou_batch = event_iou_active_only_from_class_indices(
                pred_class_idx=pred_max_idx.detach().cpu().numpy(),
                true_class_idx=true_max_idx.detach().cpu().numpy(),
            )
            active_windows_batch = np.logical_or(
                pred_max_idx.detach().cpu().numpy() > 0,
                true_max_idx.detach().cpu().numpy() > 0,
            ).any(axis=1)
            n_active_batch = int(np.sum(active_windows_batch))
            event_intersection += float(mean_iou_batch) * float(n_active_batch)
            event_union += int(n_active_batch)

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
    active_mask = support[1:] > 0
    mean_f1 = (
        float(
            np.mean(
                [f1_scores[i + 1] for i, active in enumerate(active_mask) if active]
            )
        )
        if np.any(active_mask)
        else 0.0
    )

    mean_iou = float(event_intersection / event_union) if event_union > 0 else 0.0
    mean_val_loss = (
        float(val_loss_sum / val_batch_count) if val_batch_count > 0 else 0.0
    )

    result = (
        list(f1_scores),
        mean_f1,
        mean_iou,
    )
    if return_val_loss:
        result = (*result, mean_val_loss)
    if return_event_plot_payloads:
        result = (*result, event_plot_payloads)
    if return_cm:
        return (*result, cm)
    return result


AVAILABLE_TRACE_DESCRIPTORS = (
    "envelope",
    "dominant_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
)


def _normalize_descriptor_names(
    descriptor_names: Sequence[str] | str | None,
) -> tuple[str, ...]:
    if descriptor_names is None:
        return tuple()
    if isinstance(descriptor_names, str):
        if descriptor_names.lower() == "all":
            return tuple(AVAILABLE_TRACE_DESCRIPTORS)
        names = (descriptor_names,)
    else:
        names = tuple(descriptor_names)

    unknown = sorted(set(names) - set(AVAILABLE_TRACE_DESCRIPTORS))
    if len(unknown) > 0:
        raise ValueError(
            "Unknown descriptor names: "
            f"{unknown}. Available: {list(AVAILABLE_TRACE_DESCRIPTORS)}"
        )
    return tuple(names)


def _station_permutation(
    n_stations: int,
    idx: int,
    base_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(base_seed) + int(idx))
    return rng.permutation(int(n_stations)).astype(np.int64, copy=False)


def _permute_station_tensor(
    tensor: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    result = tensor
    n_stations = int(permutation.numel())
    if result.ndim >= 1 and int(result.shape[0]) == n_stations:
        result = result.index_select(0, permutation)
    if result.ndim >= 2 and int(result.shape[1]) == n_stations:
        result = result.index_select(1, permutation)
    return result


def _permute_descriptor_payload(
    descriptor_payload: dict[str, torch.Tensor],
    permutation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        key: _permute_station_tensor(value, permutation)
        for key, value in descriptor_payload.items()
    }


class MultiStation1DDataset(Dataset):
    """Simple multi-station 1D dataset for PhaseNet-style models."""

    def __init__(
        self,
        npz_path: Path,
        scramble_stations: bool = False,
        station_scramble_seed: int = 42,
    ):
        with np.load(npz_path) as data:
            self.filepaths = data["filepaths"].copy()
            self.labels = data["labels"].copy()
            self.label_ids = data["label_ids"].astype(np.int64, copy=True)

        self.scramble_stations = bool(scramble_stations)
        self.station_scramble_seed = int(station_scramble_seed)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        arr = np.load(self.filepaths[idx], mmap_mode="r")
        arr = np.array(arr, dtype=np.float32)
        x_raw = arr[:8, :]
        y_raw = arr[8:, :]

        if self.scramble_stations:
            permutation = _station_permutation(
                n_stations=x_raw.shape[0],
                idx=idx,
                base_seed=self.station_scramble_seed,
            )
            x_raw = x_raw[permutation, :]

        x = torch.from_numpy(x_raw)
        y_onehot = torch.from_numpy(y_raw)
        y_label = torch.tensor(int(self.label_ids[idx]), dtype=torch.long)
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

    AVAILABLE_DESCRIPTORS = AVAILABLE_TRACE_DESCRIPTORS

    def __init__(
        self,
        npz_path: Path,
        descriptor_names: Sequence[str] | str | None = None,
        return_volcano_idx: bool = True,
        volcano_name_to_idx: Optional[dict[str, int]] = None,
        scramble_stations: bool = False,
        station_scramble_seed: int = 42,
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

        self.descriptor_names = _normalize_descriptor_names(descriptor_names)
        self.use_descriptors = len(self.descriptor_names) > 0
        self.return_volcano_idx = bool(return_volcano_idx)
        self.volcano_name_to_idx = (
            dict(volcano_name_to_idx) if volcano_name_to_idx is not None else {}
        )
        self.scramble_stations = bool(scramble_stations)
        self.station_scramble_seed = int(station_scramble_seed)

        if self.manifest_volcano_idx is not None:
            self.sample_volcano_idx = self.manifest_volcano_idx
        elif self.return_volcano_idx and volcano_name_to_idx is not None:
            from .station_info import infer_volcano_name_from_path

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
            self.descriptor_paths = self._infer_descriptor_paths()

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

        permutation_t: torch.Tensor | None = None
        if self.scramble_stations:
            permutation = _station_permutation(
                n_stations=x_raw.shape[0],
                idx=idx,
                base_seed=self.station_scramble_seed,
            )
            x_raw = x_raw[permutation, :]
            permutation_t = torch.as_tensor(permutation, dtype=torch.long)

        x = torch.from_numpy(x_raw)
        y_onehot = torch.from_numpy(y_raw)
        y_label = torch.tensor(int(self.label_ids[idx]), dtype=torch.long)
        output = [x, y_onehot, y_label]
        if self.use_descriptors:
            descriptors = self._load_selected_descriptors(idx)
            if permutation_t is not None:
                descriptors = _permute_descriptor_payload(
                    descriptors,
                    permutation_t,
                )
            output.append(descriptors)
        if self.return_volcano_idx:
            output.append(
                torch.tensor(int(self.sample_volcano_idx[idx]), dtype=torch.long)
            )
        return tuple(output) if len(output) > 3 else (x, y_onehot, y_label)


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


def event_vs_bg_f1_from_confusion_matrix(confusion_matrix: np.ndarray) -> float:
    """Backward-compatible wrapper around shared event-vs-BG F1."""
    return event_f1_agnostic_from_confusion_matrix(confusion_matrix)


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
    """Backward-compatible wrapper around shared scalar summaries."""
    return summarize_scalar_values(values)


def ensure_fold_data_exists(fold_data_dir: Path) -> None:
    needed = [
        fold_data_dir / "train_aug.npz",
        fold_data_dir / "val.npz",
        fold_data_dir / "test.npz",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing fold manifest files:\n" + "\n".join(missing))
