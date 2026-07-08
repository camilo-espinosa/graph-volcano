from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn

# This script uses a model implementation from the following GitHub repository:
# U-Net for brain segmentation: https://github.com/mateuszbuda/brain-segmentation-pytorch

# The model is utilized as-is from the repository and follows the
# original documentation and usage guidelines.


class UNetBottleneckAttention(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        init_features=32,
        depth=4,
        bottleneck_attn_heads=4,
        bottleneck_attn_dropout=0.2,
        bottleneck_attn_ff_mult=2,
        feature_dropout=0.2,
    ):
        super().__init__()

        if feature_dropout < 0.0 or feature_dropout >= 1.0:
            raise ValueError(
                f"feature_dropout must be in [0, 1). Got: {feature_dropout}."
            )
        self.feature_dropout_p = float(feature_dropout)

        features = init_features
        self.encoder_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        self.upconv_list = nn.ModuleList()

        feat_in = in_channels
        for idx in range(depth):
            feat_out = features * 2 ** (idx)
            encoder = self._block(
                feat_in,
                feat_out,
                name=f"enc{idx}",
                feature_dropout=self.feature_dropout_p,
            )
            feat_in = feat_out
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder_list.append(encoder)
            self.pool_list.append(pool)

        self.bottleneck_channels = features * 2**depth
        self.bottleneck = self._block(
            features * 2 ** (depth - 1),
            features * 2**depth,
            name="bottleneck",
            feature_dropout=self.feature_dropout_p,
        )

        if self.bottleneck_channels % bottleneck_attn_heads != 0:
            raise ValueError(
                "bottleneck channels must be divisible by bottleneck_attn_heads. "
                f"Got C={self.bottleneck_channels}, heads={bottleneck_attn_heads}."
            )

        self.bottleneck_attn_norm1 = nn.LayerNorm(self.bottleneck_channels)
        self.bottleneck_attn = nn.MultiheadAttention(
            embed_dim=self.bottleneck_channels,
            num_heads=bottleneck_attn_heads,
            dropout=bottleneck_attn_dropout,
            batch_first=True,
        )
        self.bottleneck_attn_norm2 = nn.LayerNorm(self.bottleneck_channels)
        self.bottleneck_ff = nn.Sequential(
            nn.Linear(
                self.bottleneck_channels,
                self.bottleneck_channels * bottleneck_attn_ff_mult,
            ),
            nn.GELU(),
            nn.Dropout(bottleneck_attn_dropout),
            nn.Linear(
                self.bottleneck_channels * bottleneck_attn_ff_mult,
                self.bottleneck_channels,
            ),
            nn.Dropout(bottleneck_attn_dropout),
        )

        for idx in range(depth):
            feat_in = features * 2 ** (depth - idx)
            feat_out = features * 2 ** (depth - 1 - idx)

            upconv = nn.ConvTranspose2d(feat_in, feat_out, kernel_size=2, stride=2)
            self.upconv_list.append(upconv)

            decoder = self._block(
                feat_in,
                feat_out,
                name=f"dec{depth-idx}",
                feature_dropout=self.feature_dropout_p,
            )
            self.decoder_list.append(decoder)

        self.final_dropout = nn.Dropout(self.feature_dropout_p)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def _apply_bottleneck_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        x_seq = x.flatten(2).transpose(1, 2)

        x_norm = self.bottleneck_attn_norm1(x_seq)
        x_attn, _ = self.bottleneck_attn(x_norm, x_norm, x_norm, need_weights=False)
        x_seq = x_seq + x_attn
        x_seq = x_seq + self.bottleneck_ff(self.bottleneck_attn_norm2(x_seq))

        return x_seq.transpose(1, 2).reshape(batch_size, channels, height, width)

    def forward(self, x):
        encodings = []
        for i in range(len(self.encoder_list)):
            x = self.encoder_list[i](x)
            encodings.append(x)
            if i < len(self.pool_list):  # Avoid index error
                x = self.pool_list[i](x)

        x = self.bottleneck(x)
        x = self._apply_bottleneck_attention(x)

        for i in range(len(self.decoder_list)):
            x = self.upconv_list[i](x)
            x = torch.cat((x, encodings[-(i + 1)]), dim=1)
            x = self.decoder_list[i](x)
        x = self.final_dropout(x)
        return self.conv(x)

    @staticmethod
    def _block(in_channels, features, name, feature_dropout):
        dropout = (
            nn.Dropout(feature_dropout) if feature_dropout > 0.0 else nn.Identity()
        )
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "drop1", dropout),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "drop2", dropout),
                ]
            )
        )
