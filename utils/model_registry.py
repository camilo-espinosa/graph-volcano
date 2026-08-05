"""Central registry for the active model definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from models.PhaseNet import PhaseNet
from models.PhaseNet_bottleneck_attention import PhaseNetBottleneckAttention
from models.MuSSeg import MuSSeg
from models.MuSSED import MuSSED
from models.UNet import UNet
from models.UNet_bottleneck_attention import UNetBottleneckAttention

PHASENET_BASE_KWARGS: dict[str, Any] = {
    "in_channels": 8,
    "classes": 6,
    "depth": 5,
    "kernel_size": 7,
    "stride": 2,
    "filters_root": 32,
    "norm": "std",
    "feature_dropout": 0.2,
}

PHASENET_BOTTLENECK_ATTENTION_BASE_KWARGS: dict[str, Any] = {
    **PHASENET_BASE_KWARGS,
    "bottleneck_attn_heads": 4,
    "bottleneck_attn_dropout": 0.2,
    "bottleneck_attn_ff_mult": 2,
}

PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS: dict[str, Any] = {
    **PHASENET_BASE_KWARGS,
    "filters_root": 32,
    "bottleneck_attention": False,
    "shared_station_encoder": False,
    "station_interaction": "none",
    "station_message_levels": [],
    "station_message_aggregation": "sum",
    "station_message_ratio": 1.0,
    "station_attention_levels": [],
    "pre_bottleneck_station_attn_merge": False,
    "bottleneck_attn_heads": 4,
    "bottleneck_attn_dropout": 0.2,
    "bottleneck_attn_ff_mult": 2,
    "station_attn_heads": 4,
    "station_attn_dropout": 0.2,
    "station_attn_ff_mult": 2,
}

MUSSEG_PI_SE_LSA_BA_BASE_KWARGS: dict[str, Any] = deepcopy(
    PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS
)
MUSSEG_PI_SE_LSA_BA_BASE_KWARGS.pop("shared_station_encoder", None)
MUSSEG_PI_SE_LSA_BA_BASE_KWARGS.update(
    {
        "bottleneck_attention": True,
        "station_interaction": "late_attention",
        "volcano_name": "NVCHVC",
    }
)

MUSSED_BASE_KWARGS: dict[str, Any] = {
    "num_classes": 6,
    "depth": 4,
    "kernel_size": 127,
    "stride": [2, 2, 2],
    "dilation": [1, 1, 1, 1],
    "filters_root": 16,
    "bottleneck_attention": True,
    "bottleneck_attn_heads": 4,
    "bottleneck_attn_ff_mult": 2,
    "station_attn_heads": 4,
    "station_attn_ff_mult": 2,
    "station_mask_abs_sum_threshold": 1e1,
    "num_queries": 10,
    "query_dim": 128,
    "hidden_dim": 256,
    "num_decoder_heads": 4,
    "num_decoder_layers": 2,
    "decoder_dropout": 0.01,
    "use_temporal_projection": False,
    "interval_output_format": "center_duration",
}

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # ===================================================================
    # EVENT DETECTION: MuSSED models (DETR-based event detection)
    # Model-search set: keep encoder fixed, vary decoder/query side only.
    # ===================================================================
    "mussed_multilevel_memory_all": {
        "family": "detr",
        "trainer_kind": "detr",
        "model_cls": MuSSED,
        "model_kwargs": {
            **deepcopy(MUSSED_BASE_KWARGS),
            "use_temporal_projection": True,
            "memory_levels": [0, 1, 2],
            "eval_memory_level_pool_to": 375,
        },
        "batch_size": 16,
    },
    "mussed_search_dec1_q8": {
        "family": "detr",
        "trainer_kind": "detr",
        "model_cls": MuSSED,
        "model_kwargs": {
            **deepcopy(MUSSED_BASE_KWARGS),
            "use_temporal_projection": True,
            "memory_levels": [0, 1, 2],
            "eval_memory_level_pool_to": 375,
            "num_decoder_layers": 1,
            "num_queries": 8,
        },
        "batch_size": 16,
    },
    "mussed_search_dec3_q8": {
        "family": "detr",
        "trainer_kind": "detr",
        "model_cls": MuSSED,
        "model_kwargs": {
            **deepcopy(MUSSED_BASE_KWARGS),
            "use_temporal_projection": True,
            "memory_levels": [0, 1, 2],
            "eval_memory_level_pool_to": 375,
            "num_decoder_layers": 3,
            "num_queries": 8,
        },
        "batch_size": 16,
    },
    "mussed_search_q6_fast": {
        "family": "detr",
        "trainer_kind": "detr",
        "model_cls": MuSSED,
        "model_kwargs": {
            **deepcopy(MUSSED_BASE_KWARGS),
            "use_temporal_projection": True,
            "memory_levels": [0, 1, 2],
            "eval_memory_level_pool_to": 375,
            "num_queries": 6,
        },
        "batch_size": 16,
    },
    "mussed_search_q14_capacity": {
        "family": "detr",
        "trainer_kind": "detr",
        "model_cls": MuSSED,
        "model_kwargs": {
            **deepcopy(MUSSED_BASE_KWARGS),
            "use_temporal_projection": True,
            "memory_levels": [0, 1, 2],
            "eval_memory_level_pool_to": 375,
            "num_queries": 14,
        },
        "batch_size": 16,
    },
    "mussed_search_ffn192": {
        "family": "detr",
        "trainer_kind": "detr",
        "model_cls": MuSSED,
        "model_kwargs": {
            **deepcopy(MUSSED_BASE_KWARGS),
            "use_temporal_projection": True,
            "memory_levels": [0, 1, 2],
            "eval_memory_level_pool_to": 375,
            "hidden_dim": 192,
        },
        "batch_size": 16,
    },
    "mussed_search_ffn384": {
        "family": "detr",
        "trainer_kind": "detr",
        "model_cls": MuSSED,
        "model_kwargs": {
            **deepcopy(MUSSED_BASE_KWARGS),
            "use_temporal_projection": True,
            "memory_levels": [0, 1, 2],
            "eval_memory_level_pool_to": 375,
            "hidden_dim": 384,
        },
        "batch_size": 16,
    },
    "mussed_search_heads8": {
        "family": "detr",
        "trainer_kind": "detr",
        "model_cls": MuSSED,
        "model_kwargs": {
            **deepcopy(MUSSED_BASE_KWARGS),
            "use_temporal_projection": True,
            "memory_levels": [0, 1, 2],
            "eval_memory_level_pool_to": 375,
            "num_decoder_heads": 8,
            "num_queries": 8,
        },
        "batch_size": 16,
    },
    "mussed_search_dropout005": {
        "family": "detr",
        "trainer_kind": "detr",
        "model_cls": MuSSED,
        "model_kwargs": {
            **deepcopy(MUSSED_BASE_KWARGS),
            "use_temporal_projection": True,
            "memory_levels": [0, 1, 2],
            "eval_memory_level_pool_to": 375,
            "decoder_dropout": 0.05,
        },
        "batch_size": 16,
    },
    "mussed_search_evalpool256": {
        "family": "detr",
        "trainer_kind": "detr",
        "model_cls": MuSSED,
        "model_kwargs": {
            **deepcopy(MUSSED_BASE_KWARGS),
            "use_temporal_projection": True,
            "memory_levels": [0, 1, 2],
            "eval_memory_level_pool_to": 256,
        },
        "batch_size": 16,
    },
    "mussed_d4_r16_s222_k127_no_bottleneck": {
        "family": "detr",
        "trainer_kind": "detr",
        "model_cls": MuSSED,
        "model_kwargs": {
            **deepcopy(MUSSED_BASE_KWARGS),
            "bottleneck_attention": False,
        },
        "batch_size": 16,
    },
    # # ===================================================================
    # # ACTIVE: 5-FOLD SELECTION
    # # ===================================================================
    # # ---- Tier A: winners / UNet-parity ----
    # "musseg_d4_r16_s222_k127": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 127,
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_d4_r16_s222_k127_d1248": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 127,
    #         "dilation": [1, 2, 4, 8],
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_d4_r16_s124_k127": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [1, 2, 4],
    #         "kernel_size": 127,
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 24,
    # },
    # "musseg_d4_r16_s222_k71": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 71,
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 32,
    # },
    # # ---- Tier C: kernel-size curve + efficiency sweet spot ----
    # "musseg_d4_r16_s222_k51": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 51,
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_d4_r16_s222_k31": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 31,
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_d4_r16_s222_k51_31_21_11": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": [51, 31, 21, 11],
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 32,
    # },
    # # ---- Tier D: new probes ----
    # # Kernel ceiling test: confirm k127 is the plateau, not still climbing.
    # "musseg_d4_r16_s222_k159": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 159,
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 32,
    # },
    # # Priors on the strong backbone: were distance/wskip masking a weak backbone?
    # "musseg_d4_r16_s222_k127_dist_both": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 127,
    #         "dilation": [1, 1, 1, 1],
    #         "use_distance_attn_bias": True,
    #         "use_distance_bottleneck_emb": True,
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_d4_r16_s222_k127_wskip": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 127,
    #         "dilation": [1, 1, 1, 1],
    #         "use_station_weighted_skips": True,
    #     },
    #     "batch_size": 32,
    # },
    # # ---- Tier B: baselines ----
    # "unet": {
    #     "family": "unet",
    #     "trainer_kind": "2d",
    #     "model_cls": UNet,
    #     "model_kwargs": {
    #         "in_channels": 1,
    #         "out_channels": 6,
    #         "init_features": 16,
    #         "depth": 5,
    #     },
    #     "batch_size": 32,
    # },
    # "unet_attention": {
    #     "family": "unet",
    #     "trainer_kind": "2d",
    #     "model_cls": UNetBottleneckAttention,
    #     "model_kwargs": {
    #         "in_channels": 1,
    #         "out_channels": 6,
    #         "init_features": 16,
    #         "depth": 5,
    #         "bottleneck_attn_heads": 4,
    #         "bottleneck_attn_dropout": 0.2,
    #         "bottleneck_attn_ff_mult": 2,
    #         "feature_dropout": 0.2,
    #     },
    #     "batch_size": 32,
    # },
    # "phasenet": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": PhaseNet,
    #     "model_kwargs": deepcopy(PHASENET_BASE_KWARGS),
    #     "batch_size": 64,
    # },
    # "phasenet_bottleneck_attention": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": PhaseNetBottleneckAttention,
    #     "model_kwargs": deepcopy(PHASENET_BOTTLENECK_ATTENTION_BASE_KWARGS),
    #     "batch_size": 64,
    # },
    # ===================================================================
    # INACTIVE: previously tested ablations (kept for reference)
    # ===================================================================
    # ---- stride / early-downsample variants ----
    # "musseg_d4_r16_s122": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [1, 2, 2],
    #     },
    #     "batch_size": 20,
    # },
    # "musseg_d4_r16_s112": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [1, 1, 2],
    #     },
    #     "batch_size": 12,
    # },
    # "musseg_d4_r16_s122_k51": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [1, 2, 2],
    #         "kernel_size": 51,
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 18,
    # },
    # "musseg_d4_r16_s212_k51": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 1, 2],
    #         "kernel_size": 51,
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 18,
    # },
    # # ---- kernel-size sweep (remaining points) ----
    # "musseg_d4_r16_s222_k91": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 91,
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_d4_r16_s222_k41": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 41,
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_d4_r16_s222_k21": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": [21, 21, 21, 21],
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_d4_r16_s222_k11": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": [11, 11, 11, 11],
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 30,
    # },
    # "musseg_d4_r16_s222_k3": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": [3, 3, 3, 3],
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 32,
    # },
    # # ---- dilation / increasing-kernel-with-depth variants ----
    # "musseg_d4_r16_s222_k51_d2": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 51,
    #         "dilation": [2, 2, 2, 2],
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_d4_r16_s222_k51_d1248": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": 51,
    #         "dilation": [1, 2, 4, 8],
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_d4_r16_s222_k3357_d1": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": [3, 3, 5, 7],
    #         "dilation": [1, 1, 1, 1],
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_d4_r16_s222_k3_d1122": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "depth": 4,
    #         "filters_root": 16,
    #         "stride": [2, 2, 2],
    #         "kernel_size": [3, 3, 3, 3],
    #         "dilation": [1, 1, 2, 2],
    #     },
    #     "batch_size": 32,
    # },
    # # ---- depth / root capacity sweep ----
    # "musseg_pi_se_lsa_ba_depth_6_root_8": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "filters_root": 8,
    #         "depth": 6,
    #     },
    #     "batch_size": 32,
    # },
    # "musseg_pi_se_lsa_ba_depth_3_root_24": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "filters_root": 24,
    #         "depth": 4,
    #     },
    #     "batch_size": 24,
    # },
    # "musseg_pi_se_lsa_ba_depth_4": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "filters_root": 32,
    #         "depth": 4,
    #     },
    #     "batch_size": 18,
    # },
    # "musseg_pi_se_lsa_ba_depth_6_root_16": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "filters_root": 16,
    #         "depth": 6,
    #     },
    #     "batch_size": 24,
    # },
    # "musseg_pi_se_lsa_ba_depth_3_root_48": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "filters_root": 48,
    #         "depth": 4,
    #     },
    #     "batch_size": 12,
    # },
    # "musseg_pi_se_lsa_ba_depth_4_root_48": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    #         "filters_root": 48,
    #         "depth": 4,
    #     },
    #     "batch_size": 10,
    # },
    # ---- distance / weighted-skip variants (on old backbone) ----
    # # "musseg_pi_se_lsa_ba_dist_both": {
    # #     "family": "phasenet",
    # #     "trainer_kind": "1d",
    # #     "model_cls": MuSSeg,
    # #     "model_kwargs": {
    # #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    # #         "use_distance_attn_bias": True,
    # #         "use_distance_bottleneck_emb": True,
    # #     },
    # #     "batch_size": 14,
    # # },
    # # "musseg_pi_se_lsa_ba_wskip": {
    # #     "family": "phasenet",
    # #     "trainer_kind": "1d",
    # #     "model_cls": MuSSeg,
    # #     "model_kwargs": {
    # #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    # #         "use_station_weighted_skips": True,
    # #     },
    # #     "batch_size": 14,
    # # },
    # # "musseg_pi_se_lsa_ba": {
    # #     "family": "phasenet",
    # #     "trainer_kind": "1d",
    # #     "model_cls": MuSSeg,
    # #     "model_kwargs": deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    # #     "batch_size": 16,
    # # },
    # # "musseg_pi_se_lsa_ba_dist_attn": {
    # #     "family": "phasenet",
    # #     "trainer_kind": "1d",
    # #     "model_cls": MuSSeg,
    # #     "model_kwargs": {
    # #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    # #         "use_distance_attn_bias": True,
    # #     },
    # #     "batch_size": 16,
    # # },
    # # "musseg_pi_se_lsa_ba_dist_emb": {
    # #     "family": "phasenet",
    # #     "trainer_kind": "1d",
    # #     "model_cls": MuSSeg,
    # #     "model_kwargs": {
    # #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    # #         "use_distance_bottleneck_emb": True,
    # #     },
    # #     "batch_size": 16,
    # # },
    # # "musseg_pi_se_lsa_ba_dist_attn_wskip": {
    # #     "family": "phasenet",
    # #     "trainer_kind": "1d",
    # #     "model_cls": MuSSeg,
    # #     "model_kwargs": {
    # #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    # #         "use_distance_attn_bias": True,
    # #         "use_station_weighted_skips": True,
    # #     },
    # #     "batch_size": 16,
    # # },
    # # "musseg_pi_se_lsa_ba_dist_emb_wskip": {
    # #     "family": "phasenet",
    # #     "trainer_kind": "1d",
    # #     "model_cls": MuSSeg,
    # #     "model_kwargs": {
    # #         **deepcopy(MUSSEG_PI_SE_LSA_BA_BASE_KWARGS),
    # #         "use_distance_bottleneck_emb": True,
    # #         "use_station_weighted_skips": True,
    # #     },
    # #     "batch_size": 16,
    # # },
    # ---- station-interaction / merge variants ----
    # "musseg_pi_se_ba": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
    #         "bottleneck_attention": True,
    #         "shared_station_encoder": True,
    #     },
    #     "batch_size": 20,
    # },
    # "musseg_pi_se_lsm_sum_ba": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
    #         "bottleneck_attention": True,
    #         "shared_station_encoder": True,
    #         "station_interaction": "late_station_message",
    #         "station_message_aggregation": "sum",
    #         "station_message_ratio": 0.25,
    #     },
    #     "batch_size": 12,
    # },
    # "musseg_pi_se_lsm_attn_ba": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
    #         "bottleneck_attention": True,
    #         "shared_station_encoder": True,
    #         "station_interaction": "late_station_message",
    #         "station_message_aggregation": "attention",
    #         "station_message_ratio": 0.25,
    #     },
    #     "batch_size": 10,
    # },
    # "musseg_pi_se_ba_pbam": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
    #         "bottleneck_attention": True,
    #         "shared_station_encoder": True,
    #         "pre_bottleneck_station_attn_merge": True,
    #     },
    #     "batch_size": 20,
    # },
    # "musseg_pi_se_lsa_ba_pbam": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
    #         "bottleneck_attention": True,
    #         "shared_station_encoder": True,
    #         "station_interaction": "late_attention",
    #         "pre_bottleneck_station_attn_merge": True,
    #     },
    #     "batch_size": 18,
    # },
    # "musseg_pi_se_lsm_sum_ba_pbam": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
    #         "bottleneck_attention": True,
    #         "shared_station_encoder": True,
    #         "station_interaction": "late_station_message",
    #         "station_message_aggregation": "sum",
    #         "station_message_ratio": 0.25,
    #         "pre_bottleneck_station_attn_merge": True,
    #     },
    #     "batch_size": 14,
    # },
    # "musseg_pi_se_lsm_attn_ba_pbam": {
    #     "family": "phasenet",
    #     "trainer_kind": "1d",
    #     "model_cls": MuSSeg,
    #     "model_kwargs": {
    #         **deepcopy(PHASENET_PERMUTATION_INVARIANT_BASE_KWARGS),
    #         "bottleneck_attention": True,
    #         "shared_station_encoder": True,
    #         "station_interaction": "late_station_message",
    #         "station_message_aggregation": "attention",
    #         "station_message_ratio": 0.25,
    #         "pre_bottleneck_station_attn_merge": True,
    #     },
    #     "batch_size": 12,
    # },
}


def get_model_spec(model_key: str) -> dict[str, Any]:
    if model_key in MODEL_REGISTRY:
        return deepcopy(MODEL_REGISTRY[model_key])
    raise KeyError(
        f"Unknown model key '{model_key}'. Available: {sorted(MODEL_REGISTRY.keys())}"
    )


def list_model_specs(
    *,
    family: str | None = None,
    trainer_kind: str | None = None,
    preserve_order: bool = False,
) -> dict[str, dict[str, Any]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for key, spec in MODEL_REGISTRY.items():
        if family is not None and spec["family"] != family:
            continue
        if trainer_kind is not None and spec["trainer_kind"] != trainer_kind:
            continue
        items.append((key, deepcopy(spec)))
    if not preserve_order:
        items.sort(key=lambda item: (item[1]["family"], item[0]))
    return dict(items)


def build_model_from_spec(model_key: str, n_classes: int = 6, **overrides):
    spec = get_model_spec(model_key)
    model_cls = spec["model_cls"]
    kwargs = dict(spec["model_kwargs"])
    kwargs.update(overrides)
    if spec["family"] == "phasenet":
        kwargs.setdefault("classes", n_classes)
        kwargs.setdefault("in_channels", 8)
    elif spec["family"] == "unet":
        kwargs.setdefault("out_channels", n_classes)
        kwargs.setdefault("in_channels", 1)
    else:
        pass
    return model_cls(**kwargs)


MODEL_SPECS = {
    key: {
        "model_cls": spec["model_cls"],
        "model_kwargs": deepcopy(spec["model_kwargs"]),
        "family": spec["family"],
        "trainer_kind": spec["trainer_kind"],
        "batch_size": spec["batch_size"],
    }
    for key, spec in MODEL_REGISTRY.items()
}
