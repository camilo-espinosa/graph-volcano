import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class Feedforward(nn.Module):
    def __init__(self, channel_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(channel_size * 2 + 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, channel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 2, 1)


class FeedforwardAggr(nn.Module):
    def __init__(self, channel_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(channel_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, channel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.permute(0, 2, 1)


class GNO(MessagePassing):
    def __init__(self, in_channels: int):
        super().__init__(aggr="mean", node_dim=-3)
        hidden = in_channels * 4
        self.edge = Feedforward(in_channels, hidden)
        self.aggr = FeedforwardAggr(in_channels, hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(
            edge_index[:2, :].to(torch.int64),
            x=x,
            xloc=edge_index[2:, :],
        )

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, xloc: torch.Tensor
    ) -> torch.Tensor:
        nedge = xloc.shape[1]
        loc_i = torch.zeros(
            (nedge, 2, x_i.shape[-1]), device=x_i.device, dtype=x_i.dtype
        )
        loc_j = torch.zeros(
            (nedge, 2, x_j.shape[-1]), device=x_j.device, dtype=x_j.dtype
        )

        for e in range(nedge):
            loc_i[e, 0, :] = xloc[0, e]
            loc_i[e, 1, :] = xloc[1, e]
            loc_j[e, 0, :] = xloc[2, e]
            loc_j[e, 1, :] = xloc[3, e]

        tmp = torch.cat([x_i, loc_i, x_j, loc_j], dim=1)
        return self.edge(tmp)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tmp = torch.cat([aggr_out, x], dim=1)
        return self.aggr(tmp)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dim1: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.dim1 = dim1
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand((in_channels, out_channels, self.modes1), dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor, dim1: int | None = None) -> torch.Tensor:
        if dim1 is not None:
            self.dim1 = dim1

        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x, norm="forward")

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            self.dim1 // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1],
            self.weights1,
        )

        return torch.fft.irfft(out_ft, n=self.dim1, norm="forward")


class PointwiseOp1D(nn.Module):
    def __init__(self, in_codim: int, out_codim: int, dim1: int):
        super().__init__()
        self.conv = nn.Conv1d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)

    def forward(self, x: torch.Tensor, dim1: int | None = None) -> torch.Tensor:
        if dim1 is None:
            dim1 = self.dim1
        x_out = self.conv(x)
        return F.interpolate(x_out, size=dim1, mode="linear", align_corners=True)


class FNO1D(nn.Module):
    def __init__(
        self,
        in_codim: int,
        out_codim: int,
        modes1: int,
        dim1: int,
        normalize: bool = True,
        non_lin: bool = True,
    ):
        super().__init__()
        self.conv = SpectralConv1d(in_codim, out_codim, int(dim1), int(modes1))
        self.w = PointwiseOp1D(in_codim, out_codim, int(dim1))
        self.normalize = normalize
        self.non_lin = non_lin
        if normalize:
            self.normalize_layer = nn.InstanceNorm1d(int(out_codim), affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.conv(x) + self.w(x)
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


class PhaseNOCore(nn.Module):
    def __init__(
        self,
        in_features: int = 5,
        out_features: int = 2,
        modes: int = 24,
        width: int = 48,
        padding: int = 50,
        dim_mid_1: int = 750,
        dim_mid_2: int = 200,
    ):
        super().__init__()
        self.modes1 = int(modes)
        self.width = int(width)
        self.padding = int(padding)

        self.fc0 = nn.Linear(in_features + 1, self.width)

        long_dim = 3000 + self.padding * 2
        self.fno0 = FNO1D(self.width, self.width, self.modes1, long_dim)
        self.gno0 = GNO(self.width)

        self.fno1 = FNO1D(self.width, self.width * 2, self.modes1 // 2, dim_mid_1)
        self.gno1 = GNO(self.width * 2)

        self.fno2 = FNO1D(
            self.width * 2, self.width * 4, max(1, self.modes1 // 3), dim_mid_2
        )
        self.gno3 = GNO(self.width * 4)

        self.fno4 = FNO1D(
            self.width * 8, self.width * 2, max(1, self.modes1 // 3), dim_mid_1
        )
        self.gno4 = GNO(self.width * 2)

        self.fno5 = FNO1D(self.width * 4, self.width, self.modes1 // 2, long_dim)
        self.gno5 = GNO(self.width)

        self.fno6 = FNO1D(self.width * 2, self.width, self.modes1, long_dim)
        self.fno7 = FNO1D(self.width, self.width, self.modes1, long_dim, non_lin=False)

        self.fc1 = nn.Linear(self.width, self.width * 2)
        self.fc2 = nn.Linear(self.width * 2, out_features)

    @staticmethod
    def _grid(
        shape: torch.Size, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        nstation, size_t = shape[-3], shape[-1]
        gridx = torch.linspace(0.0, 1.0, steps=size_t, device=device, dtype=dtype)
        return gridx.reshape(1, 1, size_t).repeat(nstation, 1, 1)

    def forward(self, data: tuple[torch.Tensor, None, torch.Tensor]) -> torch.Tensor:
        x, edge_index = data[0], data[2]

        grid = self._grid(x.shape, x.device, x.dtype)
        x = torch.cat((x, grid), dim=1)
        x = F.pad(x, [self.padding, self.padding], mode="reflect")

        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x0 = self.fno0(x)
        x = self.gno0(x0, edge_index)

        x1 = self.fno1(x)
        x = self.gno1(x1, edge_index)

        x2 = self.fno2(x)
        x = self.gno3(x2, edge_index)
        x = torch.cat([x2, x], dim=1)

        x = self.fno4(x)
        x = self.gno4(x, edge_index)
        x = torch.cat([x1, x], dim=1)

        x = self.fno5(x)
        x = self.gno5(x, edge_index)
        x = torch.cat([x0, x], dim=1)

        x = self.fno6(x)
        x = self.fno7(x)

        x = x[..., self.padding : -self.padding]

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x.permute(0, 2, 1)


class PhaseNO(nn.Module):
    """
    Adapter with PhaseNet-compatible interface:
    input [B, S, T] -> output logits [B, C, T].
    """

    def __init__(
        self,
        in_channels: int = 8,
        classes: int = 6,
        modes: int = 24,
        width: int = 48,
        edge_distance_threshold: float | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.classes = int(classes)
        self.edge_distance_threshold = edge_distance_threshold

        self.core = PhaseNOCore(
            in_features=5,
            out_features=self.classes,
            modes=modes,
            width=width,
        )

        station_xy = self._default_station_xy(self.in_channels)
        self.register_buffer("station_xy", station_xy, persistent=True)
        self.register_buffer(
            "edge_index_template",
            self._build_edge_index(station_xy, edge_distance_threshold),
            persistent=True,
        )

    @staticmethod
    def _default_station_xy(n_stations: int) -> torch.Tensor:
        if n_stations <= 1:
            return torch.zeros((n_stations, 2), dtype=torch.float32)
        x = torch.linspace(0.0, 1.0, steps=n_stations)
        y = torch.zeros_like(x)
        return torch.stack([x, y], dim=1)

    @staticmethod
    def _build_edge_index(
        station_xy: torch.Tensor,
        distance_threshold: float | None,
    ) -> torch.Tensor:
        row_a, row_b, row_ix, row_iy, row_jx, row_jy = [], [], [], [], [], []
        n_stations = station_xy.shape[0]

        for i in range(n_stations):
            for j in range(n_stations):
                if distance_threshold is not None:
                    dist = torch.norm(station_xy[i] - station_xy[j]).item()
                    if dist > float(distance_threshold):
                        continue
                row_a.append(i)
                row_b.append(j)
                row_ix.append(station_xy[i, 0].item())
                row_iy.append(station_xy[i, 1].item())
                row_jx.append(station_xy[j, 0].item())
                row_jy.append(station_xy[j, 1].item())

        edge_index = np.array(
            [row_a, row_b, row_ix, row_iy, row_jx, row_jy], dtype=np.float32
        )
        return torch.from_numpy(edge_index)

    def _expand_to_phaseno_features(
        self, station_waveforms: torch.Tensor
    ) -> torch.Tensor:
        s, t = station_waveforms.shape
        x = torch.zeros(
            (s, 5, t), device=station_waveforms.device, dtype=station_waveforms.dtype
        )
        x[:, 0, :] = station_waveforms
        x[:, 3, :] = self.station_xy[:, 0].unsqueeze(-1)
        x[:, 4, :] = self.station_xy[:, 1].unsqueeze(-1)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.ndim == 4 and x.shape[2] == 1:
            x = x[:, :, 0, :]
        if x.ndim != 3:
            raise ValueError(
                "PhaseNO expects input shape [B, S, T] or [B, S, 1, T]. "
                f"Got: {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected S=in_channels={self.in_channels} stations, got S={x.shape[1]}."
            )

        edge_index = self.edge_index_template.to(device=x.device, dtype=x.dtype)

        out_batch = []
        for b in range(x.shape[0]):
            station_waveforms = x[b]
            phaseno_in = self._expand_to_phaseno_features(station_waveforms)
            station_logits = self.core((phaseno_in, None, edge_index))
            sample_logits = station_logits.mean(dim=0)
            out_batch.append(sample_logits)

        return torch.stack(out_batch, dim=0)
