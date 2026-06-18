import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import gc
import pandas as df
from torch.utils.data import DataLoader, random_split
from torch import optim
import time
import importlib.util
import segmentation_models_pytorch as smp
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LEGACY_SWIN_DIR = ROOT_DIR.parent / "volcano-seismic-segmentation-main" / "models"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.UNet import UNet
from models.PhaseNet import PhaseNet


def load_legacy_swin_transformer():
    swin_path = LEGACY_SWIN_DIR / "SwinUNet.py"
    spec = importlib.util.spec_from_file_location("legacy_swinunet", swin_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SwinUNet module from {swin_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwinTransformerSys


SwinTransformerSys = load_legacy_swin_transformer()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def count_trainable_parameters(model: nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")


class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._devnull = open(os.devnull, "w")
        sys.stdout = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        if not self._devnull.closed:
            self._devnull.close()


def free_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def print_time(t_i, t_f):
    elapsed_time_seconds = t_f - t_i
    hours = int(elapsed_time_seconds // 3600)
    minutes = int((elapsed_time_seconds % 3600) // 60)
    seconds = int(elapsed_time_seconds % 60)
    print("Elapsed time: {:02d}:{:02d}:{:02d}".format(hours, minutes, seconds))


def f1_score_from_confusion_matrix(confusion_matrix_):
    f1_scores = []
    for i in range(confusion_matrix_.shape[0]):
        tp = confusion_matrix_[i, i]
        fp = np.sum(confusion_matrix_[:, i]) - tp
        fn = np.sum(confusion_matrix_[i, :]) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        f1_scores.append(f1)
    return f1_scores


def dice_loss_2D(pred, target):
    smooth = 1.0
    iflat = pred.contiguous().view(-1)
    iflat = iflat / iflat.max()
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    a_sum = torch.sum(iflat)
    b_sum = torch.sum(tflat)
    return 1 - ((2.0 * intersection + smooth) / (a_sum + b_sum + smooth))


def model_selector(arch, N=256):
    if arch == "UNet_e":
        return UNet(in_channels=1, out_channels=6, init_features=16, depth=5).to(device)
    if arch == "UNetPlusPlus_x":
        return smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b1",
            in_channels=1,
            classes=6,
            encoder_weights=None,
            activation="softmax2d",
            decoder_channels=[64, 32, 16],
            encoder_depth=3,
        ).to(device)
    if arch == "DeepLab_y":
        return smp.DeepLabV3Plus(
            encoder_depth=5,
            decoder_channels=128,
            encoder_name="mobilenet_v2",
            in_channels=1,
            classes=6,
            encoder_weights="imagenet",
            activation="softmax2d",
        ).to(device)
    if arch == "SwinUNet_z":
        return SwinTransformerSys(
            img_size=N,
            patch_size=4,
            in_chans=1,
            num_classes=6,
            embed_dim=64,
            depths=[2, 2, 2, 2],
            depths_decoder=[1, 2, 2, 2],
            num_heads=[2, 4, 8, 16],
            window_size=8,
        ).to(device)
    if arch == "PhaseNet_B":
        return PhaseNet(
            in_channels=8,
            classes=6,
            depth=5,
            kernel_size=7,
            stride=2,
            norm="std",
            filters_root=32,
        ).to(device)
    raise ValueError(f"Unsupported architecture: {arch}")


class CustomTrace2DDataset(Dataset):
    def __init__(self, data_info_pd, len_window=9800, n_classes=6, im_size=280):
        comb_list = [
            [392, 56],
            [1568, 112],
            [3528, 168],
            [6272, 224],
            [9800, 280],
            [14112, 336],
        ]
        n = np.sqrt(8 * len_window)
        assert n == im_size, (
            "trace length and im_size do not match. For 8 channels and window of 7, "
            f"please choose one of the following combinations: {comb_list}"
        )
        self.data_info = data_info_pd
        self.len_window = len_window
        self.n_classes = n_classes
        self.im_size = im_size

    def __len__(self):
        return len(self.data_info["true_label"])

    def __getitem__(self, idx):
        path = self.data_info.loc[idx]["event_path"]
        event_name = self.data_info.loc[idx]["event_name"]
        data_input = np.load(path)
        data_input = torch.tensor(data_input.copy())
        X = data_input[:8].float()
        y = data_input[8 : 9 + self.n_classes].float()
        patches = X.unfold(1, self.im_size, self.im_size)
        patches = patches.permute(1, 0, 2)
        X = patches.reshape(-1, self.im_size).unsqueeze(0)
        patches = y.repeat(8, 1, 1)
        patches = patches.permute(1, 0, 2)
        patches = patches.unfold(2, self.im_size, self.im_size)
        patches = patches.permute(0, 2, 1, 3)
        y = patches.reshape(self.n_classes + 1, -1, self.im_size)
        return X, y, event_name


def create_dataset(
    data_dir_path,
    class_names={"VT": 1.0, "LP": 2.0, "TR": 3.0, "AV": 4.0, "IC": 5.0},
):
    data_dir = []
    for folder in class_names.keys():
        for data in os.listdir(f"{data_dir_path}/{folder}"):
            path_ = f"{data_dir_path}/{folder}/{data}"
            data_dir.append([class_names[folder], path_, data])
    return df.DataFrame(
        data=data_dir, columns=["true_label", "event_path", "event_name"]
    )


def img_to_trace_y(img, len_window=8192, im_size=256, n_classes=6):
    output = torch.zeros([len(img), n_classes, len_window])
    for idx, patches in enumerate(img):
        patches = patches.unfold(1, 8, 8)
        patches = patches.permute(0, 3, 1, 2).reshape(
            n_classes, 8, im_size * im_size // 8
        )
        patches_y = patches.sum(axis=1)
        patches_y = patches_y / patches_y.max()
        output[idx] = patches_y
    del patches_y, img
    return output


def img_to_trace_X(img, len_window=8192):
    output = torch.zeros([len(img), 8, len_window])
    for idx, patches in enumerate(img):
        patches = patches.squeeze(0)
        patches = patches.unfold(0, 8, 8)
        patches = torch.cat(patches.unbind(dim=0))
        patches = patches.permute(1, 0)
        output[idx] = patches
    del patches, img
    return output


def longest_event(BG_diff):
    start_indices = np.where(BG_diff == -1)[0]
    if len(start_indices) == 0:
        start_indices = np.array([0])
    end_indices = np.where(BG_diff == 1)[0]
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
            events.append([start, len(BG_diff) - 1, len(BG_diff) - 1 - start])
    if last_end_idx != -1:
        invalid_ends = end_indices[end_indices < start_indices[0]]
        for invalid_end in invalid_ends:
            events.insert(0, (0, invalid_end, invalid_end))
    events_df = df.DataFrame(events, columns=["start", "end", "length"])
    idx_max = events_df["length"].idxmax()
    return (
        events_df["start"][idx_max],
        events_df["end"][idx_max],
        events_df["length"][idx_max],
    )


def predicted_from_output(
    out_np,
    clases_OVDAS={1.0: "VT", 2.0: "LP", 3.0: "TR", 4.0: "AV", 5.0: "IC"},
    T_BG=50,
    T_CL=25,
):
    max_indices = np.argmax(out_np, axis=0)
    processed_out = np.eye(len(out_np))[max_indices].T
    BG_diff = np.diff(processed_out[0])
    if np.abs(BG_diff).sum() != 0:
        start_, end_, _ = longest_event(BG_diff)
    else:
        start_, end_ = 0, len(processed_out[0]) - 1
    predicted_class = processed_out[1:, start_:end_].sum(axis=1).argmax() + 1
    pred_label = clases_OVDAS[predicted_class]
    return predicted_class, pred_label, start_, end_


def cm_eval(
    model,
    dataloader,
    device,
    len_window=8192,
    im_size=256,
    clases_list={1.0: "VT", 2.0: "LP", 3.0: "TR", 4.0: "AV", 5.0: "IC"},
    T_BG=0,
    T_CL=0,
):
    print("validating...")
    pred_label = []
    true_label = []
    for _, data in enumerate(dataloader):
        X, target, _ = data
        X = X.to(device)
        target = img_to_trace_y(
            target.to(device), len_window, im_size, n_classes=len(clases_list) + 1
        )
        true_label_temp = target[:, 1:, :].sum(axis=2).max(axis=1).indices.numpy() + 1
        true_label.extend(true_label_temp.tolist())
        output = model(X)
        output = img_to_trace_y(
            output, len_window, im_size, n_classes=len(clases_list) + 1
        )
        for idx in range(len(output)):
            out_np = output[idx].detach().cpu().numpy()
            pred, _, _, _ = predicted_from_output(
                out_np, clases_list, T_BG=T_BG, T_CL=T_CL
            )
            pred_label.append(pred)
    cm = confusion_matrix(true_label, pred_label, labels=[1, 2, 3, 4, 5])
    del X, output, target, pred
    free_gpu_memory()
    return cm


def cm_save(cm, cm_path, cm_title, clases, fontsiez=12, save=True, cmap="hot_r"):
    f1_scores = f1_score_from_confusion_matrix(cm)
    mean_f1 = np.mean(f1_scores)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_percent, interpolation="nearest", cmap=cmap)
    plt.title(f"Confusion Matrix (f1:{mean_f1:.3f})", fontsize=fontsiez * 1.3333)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsiez)
    tick_marks = np.arange(len(clases))
    plt.xticks(tick_marks, clases, rotation=0, fontsize=fontsiez)
    plt.yticks(tick_marks, clases, fontsize=fontsiez)
    plt.xlabel("PREDICTED", fontsize=fontsiez * 1.166666)
    plt.ylabel("OVDAS", fontsize=fontsiez * 1.166666)
    if cm_title is not None:
        plt.title(f"{cm_title} | f1:{mean_f1:.3f}", fontsize=fontsiez * 1.166666)
    for i in range(len(clases)):
        for j in range(len(clases)):
            plt.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_percent[i, j]*100:.1f}%)",
                ha="center",
                va="center",
                color="blue",
                fontsize=fontsiez,
            )
    plt.tight_layout(pad=0)
    if save:
        plt.savefig(cm_path, bbox_inches="tight", pad_inches=0, transparent=False)
        plt.close()


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# --------- HYPERPARAMETERS --------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() == True else "cpu"
clases_OVDAS2 = {"VT": 1, "LP": 2, "TR": 3, "AV": 4, "IC": 5}

# arch = "UNet_e"  # UNet_e: 128-512 | 'UNetPlusPlus_x': 90-256 | DeepLab_y:  128-512 | SwinUNet_z: 80-256
volcano = "NVCh"


dictionarioy = {
    8192: {
        # "UNet_e": 128,
        # "UNetPlusPlus_x": 90,
        # "DeepLab_y": 128,
        "SwinUNet_z": 24,
    },
    2048: {
        # "UNet_e": 128,
        # "UNetPlusPlus_x": 90,
        # "DeepLab_y": 128,
        "SwinUNet_z": 80,
    },
    512: {
        # "UNet_e": 512,
        # "UNetPlusPlus_x": 256,
        # "DeepLab_y": 512,
        "SwinUNet_z": 256,
    },
}

W = 2048
for W in dictionarioy.keys():
    arch_dict = dictionarioy[W]
    for arch in arch_dict.keys():
        batch_size = arch_dict[arch]
        N = int(np.sqrt(W * 8))

        work_path = f"D:/Camilo/Volcanes_UFRO/RESULTADOS/PESOS_{W}"
        model_name = f"{arch}_{N}"
        # batch_size = 128
        train_size = 1500
        if W == 512:
            train_size = 2500
        epochs = 250
        lr = 1e-2
        lr_final = 1e-6
        early_stopping = EarlyStopping(patience=50, min_delta=1e-4)

        # --------------- DATASET CREATION ----------------------------------------------------
        df_train = create_dataset(
            f"D:/Camilo/Volcanes_UFRO/DATOS/{volcano}/{W}/trazas_{volcano}_{W}_train",
            clases_OVDAS2,
        )
        df_val = create_dataset(
            f"D:/Camilo/Volcanes_UFRO/DATOS/{volcano}/{W}/trazas_{volcano}_{W}_val",
            clases_OVDAS2,
        )
        df_test = create_dataset(
            f"D:/Camilo/Volcanes_UFRO/DATOS/{volcano}/{W}/trazas_{volcano}_{W}_test",
            clases_OVDAS2,
        )
        df_train = df.concat(
            [
                df_train[df_train["true_label"] == 1].sample(
                    n=train_size, random_state=42
                ),
                df_train[df_train["true_label"] == 2].sample(
                    n=train_size, random_state=42
                ),
                df_train[df_train["true_label"] == 3].sample(
                    n=train_size, random_state=42
                ),
                df_train[df_train["true_label"] == 4].sample(
                    n=train_size, random_state=42
                ),
                df_train[df_train["true_label"] == 5].sample(
                    n=train_size, random_state=42
                ),
            ]
        )
        df_train = df_train.reset_index()
        print(df_train.value_counts("true_label"))
        print(df_val.value_counts("true_label"))
        print(df_test.value_counts("true_label"))
        train_ = CustomTrace2DDataset(df_train, n_classes=5, len_window=W, im_size=N)
        val_ = CustomTrace2DDataset(df_val, n_classes=5, len_window=W, im_size=N)
        test_ = CustomTrace2DDataset(df_test, n_classes=5, len_window=W, im_size=N)

        train_loader = DataLoader(
            train_, batch_size=batch_size, shuffle=True
        )  # batch= 8: _ segundos
        val_loader = DataLoader(val_, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_, batch_size=batch_size, shuffle=True)

        # ------ Save info -------------------------------------------------------------------------
        os.makedirs(
            f"{work_path}/{model_name}/",
            exist_ok=True,
        )
        alpha = 1

        filename = f"{work_path}/{model_name}/{model_name}.txt"
        with open(filename, "w") as file:
            file.write(
                f"{model_name} \nepochs: {epochs} \nloss function: DICE \nalpha:{alpha}"
            )
            file.write(f"\nlearning rate: {lr} to {lr_final}")
            file.write(f"\nwindow size: {W} \n image size: {N}\nbatch size: {16}")
            file.write(f"\npatch size: {4} \nwindow size: {8}")

        # -----------------------MODEL SELECTION and weight loading ---------------------------------------------------------------------------
        model = model_selector(arch, N)
        initialize_weights(model)
        count_trainable_parameters(model)
        weights_path = f"{work_path}/{model_name}/{model_name}_best_f1.pt"

        try:
            loaded_model = torch.load(weights_path, weights_only=False)
            model.load_state_dict(loaded_model["model_state_dict"])
            bottom_loss = loaded_model["loss"]
            val_loss = loaded_model["loss"]
            top_f1 = loaded_model["f1score"]
            mean_f1 = loaded_model["f1score"]
            best_train_loss = 999999
        except FileNotFoundError:
            loaded_model = None
            bottom_loss = 99999999
            best_train_loss = 999999
            val_loss = 99999999
            top_f1 = 0
            mean_f1 = 0

        # ------------- Initialization ------------------------------------------------------------------------------------
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=25, eta_min=lr_final
        )
        len_data = len(train_loader)
        t_i = time.time()

        # ----------- training loop ---------------------------------------------------------------------
        metrics_list = []
        for epoch in range(epochs):
            # ---------------------------------------------------- TRAIN ----------------------------------
            with SuppressPrint():
                model.train()
            train_loss = 0.0
            for idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                X, target, name = data
                X = X.to(device)
                target = target.to(device)
                output = model(X)
                loss = dice_loss_2D(output, target)
                loss.backward()
                current_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                train_loss += loss.item()
                scheduler.step()  # Update learning rate at each step
                if idx % 50 == 0:
                    print(
                        f"loss:{loss.item():.4f}, ep: {epoch}, b: {idx+1}/{len_data}, lr {current_lr:.1e}"
                    )
            scheduler.step()
            print(f"EPOCH: {epoch} | TRAIN loss: {train_loss}")
            if train_loss < best_train_loss:
                print(
                    f"saving best train_loss {best_train_loss:.2f} -> {train_loss:.2f}"
                )
                best_train_loss = train_loss
                PATH = f"{work_path}/{model_name}/{model_name}_best_train_loss.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                        "f1score": mean_f1,
                    },
                    PATH,
                )
            # ------------------------------------------- VALIDATION ----------------------------------------
            with SuppressPrint():
                model.eval()
            with torch.no_grad():
                cm = cm_eval(model, val_loader, device, W, N)
                f1_scores = f1_score_from_confusion_matrix(cm)
                mean_f1 = np.mean(f1_scores)
            val_loss = 0.0
            with torch.no_grad():
                for idx, data in enumerate(val_loader):
                    X, target, name = data
                    X = X.to(device)
                    target = target.to(device)
                    output = model(X)
                    loss = dice_loss_2D(output, target)
                    val_loss += loss.item()
            print(f"EPOCH: {epoch} | VALIDATION loss: {val_loss}")
            print(f"EPOCH: {epoch} | VALIDATION mean_f1: {mean_f1}")
            metrics_list.append(
                [
                    current_lr,
                    epoch,
                    train_loss,
                    val_loss,
                    f1_scores[0],
                    f1_scores[1],
                    f1_scores[2],
                    f1_scores[3],
                    f1_scores[4],
                    mean_f1,
                ]
            )
            if val_loss < bottom_loss:
                print(f"saving weights {bottom_loss:.2f} -> {val_loss:.2f} loss")
                bottom_loss = val_loss
                PATH = f"{work_path}/{model_name}/{model_name}_best_val_loss.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                        "f1score": mean_f1,
                    },
                    PATH,
                )
            if mean_f1 > top_f1:
                print(f"saving weights {top_f1:.2f} -> {mean_f1:.2f} f1")
                top_f1 = mean_f1
                PATH = f"{work_path}/{model_name}/{model_name}_best_f1.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                        "f1score": mean_f1,
                    },
                    PATH,
                )
            t_f = time.time()
            print_time(t_i, t_f)
            metrics_df = df.DataFrame(
                metrics_list,
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
                ],
            )
            metrics_df.to_csv(f"{work_path}/{model_name}/{model_name}.csv")
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        metrics_df = df.DataFrame(
            metrics_list,
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
            ],
        )
        metrics_df.to_csv(f"{work_path}/{model_name}/{model_name}.csv")

        # -------------------- PLOTTING TRAINING METRICS -------------------------------
        fig, ax1 = plt.subplots()
        sns.lineplot(metrics_df, x="epoch", y="train_loss", ax=ax1, color="b")
        ax1.set_ylabel("Train Loss", color="b")
        ax2 = ax1.twinx()
        sns.lineplot(metrics_df, x="epoch", y="lr", ax=ax2, color="r")
        ax2.set_ylabel("Learning Rate (lr)", color="r")
        ax1.legend(["Train Loss"], loc="upper left")
        # ax1.set_ylim(150,250)
        ax2.legend(["Learning Rate"], loc="upper right")
        plt.savefig(f"{work_path}/{model_name}/train_loss_lr_{lr:.2e}.png")
        plt.close()
        fig, ax1 = plt.subplots()
        sns.lineplot(metrics_df, x="epoch", y="mean_f1", ax=ax1, color="b")
        ax1.set_ylabel("f1 ", color="b")
        ax2 = ax1.twinx()
        sns.lineplot(metrics_df, x="epoch", y="lr", ax=ax2, color="r")
        ax2.set_ylabel("Learning Rate (lr)", color="r")
        ax1.legend(["f1 score"], loc="upper left")
        ax2.legend(["Learning Rate"], loc="upper right")
        plt.savefig(f"{work_path}/{model_name}/f1_{lr:.1e}.png")
        plt.close()
        fig, ax1 = plt.subplots()
        sns.lineplot(metrics_df, x="epoch", y="val_loss", ax=ax1, color="b")
        ax1.set_ylabel("Validation Loss", color="b")
        ax2 = ax1.twinx()
        sns.lineplot(metrics_df, x="epoch", y="lr", ax=ax2, color="r")
        ax2.set_ylabel("Learning Rate (lr)", color="r")
        ax1.legend(["val loss"], loc="upper left")
        # ax1.set_ylim(30,100)
        ax2.legend(["Learning Rate"], loc="upper right")
        plt.savefig(f"{work_path}/{model_name}/val_loss_{lr:.1e}.png")
        plt.close()

        try:
            del model
        except:
            pass
        try:
            del optimizer
        except:
            pass
        try:
            del X
        except:
            pass
        try:
            del target
        except:
            pass
        try:
            del output
        except:
            pass
        try:
            del loss
        except:
            pass
        free_gpu_memory()

        # -------------------------- test and example predictions -------------------
        import matplotlib

        matplotlib.use("Agg")
        model = model_selector(arch, N)
        w1 = f"{work_path}/{model_name}/{model_name}_best_f1.pt"
        w2 = f"{work_path}/{model_name}/{model_name}_best_val_loss.pt"
        w3 = f"{work_path}/{model_name}/{model_name}_best_train_loss.pt"
        w_list = [w1, w2, w3]
        for weights_path in w_list:
            # weights_path = f"{work_path}/{model_name}/{model_name}_best_val_loss.pt"
            model.load_state_dict(
                torch.load(weights_path, weights_only=False)["model_state_dict"]
            )

            with SuppressPrint():
                model.eval()
                with torch.no_grad():
                    cm = cm_eval(
                        model,
                        test_loader,
                        device,
                        W,
                        N,
                        clases_list={1: "VT", 2: "LP", 3: "TR", 4: "AV", 5: "IC"},
                    )
                    cm_path = f"{work_path}/{model_name}/{model_name}_CM_{weights_path[-13:-3]}.png"
                    cm_title = f"{model_name} Confusion Matrix"
                    clases = ["VT", "LP", "TR", "AV", "IC"]
                    cm_save(
                        cm,
                        cm_path,
                        cm_title,
                        clases,
                        fontsiez=12,
                        save=True,
                        cmap="hot_r",
                    )
                    # os.makedirs(f"{work_path}/{model_name}/examples", exist_ok=True)
                    # for kdx, sample_ in enumerate(test_loader):
                    #     X_ = sample_[0].to(device)
                    #     y_ = sample_[1]
                    #     event_name_ = sample_[2]
                    #     out = model(X_)
                    #     X_ = img_to_trace_X(X_, W)
                    #     out = img_to_trace_y(out, W, N)
                    #     y_ = img_to_trace_y(y_, W, N)
                    #     for ijk in range(len(X_)):
                    #         # ijk=1
                    #         input_ = X_[ijk].squeeze(0).detach().cpu().numpy()
                    #         output = out[ijk].squeeze(0).detach().cpu().numpy()
                    #         true_label = y_[ijk]
                    #         title = event_name_[ijk]
                    #         fig, axes = plt.subplots(10, 1, sharex=True, figsize=(16, 8))
                    #         for idx, wave in enumerate(input_):
                    #             sns.lineplot(wave, ax=axes[idx])
                    #         for jdx, wave in enumerate(output):
                    #             sns.lineplot(wave, ax=axes[idx + 1])
                    #         for jdx, wave in enumerate(true_label):
                    #             sns.lineplot(wave, ax=axes[idx + 2], label=jdx)
                    #         plt.suptitle(title)
                    #         plt.savefig(f"{work_path}/{model_name}/examples/example{title}.png")
                    #         plt.close()
                    #         plt.show()
                    #     if kdx > 0:
                    #         break
        try:
            del model
        except:
            pass
        try:
            del optimizer
        except:
            pass
        try:
            del X
        except:
            pass
        try:
            del target
        except:
            pass
        try:
            del output
        except:
            pass
        try:
            del loss
        except:
            pass
        free_gpu_memory()
