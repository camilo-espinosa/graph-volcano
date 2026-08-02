"""
Integration test for MuSSED training pipeline.

Verifies that all Phase 2 and Phase 3 components work together:
- Event target conversion
- Hungarian matching
- Loss computation
- Metrics evaluation
- Trainer orchestration

Run with: python tests/test_mussed_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.MuSSED import MuSSED
from utils.event_targets import segmentation_to_events, batch_segmentation_to_events
from utils.hungarian_matcher_1d import HungarianMatcher
from utils.detr_event_loss import DETREventLoss
from utils.event_detection_metrics import EventDetectionMetrics
from utils.trainer_detr_1d import DETRTrainer


def test_event_targets():
    """Test segmentation to events conversion."""
    print("\n" + "=" * 80)
    print("TEST 1: Event Target Conversion")
    print("=" * 80)

    # Create synthetic one-hot segmentation
    # [C, T] where C=6, T=1000
    batch_size = 2
    num_classes = 6
    time_steps = 1000

    y_batch = np.zeros((batch_size, num_classes, time_steps), dtype=np.float32)

    # Add some synthetic events
    # Sample 0: VT event (class 1) from 100-200, LP event (class 2) from 300-400
    y_batch[0, 1, 100:200] = 1.0
    y_batch[0, 2, 300:400] = 1.0

    # Sample 1: Single TR event (class 3)
    y_batch[1, 3, 500:700] = 1.0

    # Convert to events
    events_batch = batch_segmentation_to_events(y_batch, normalize=True)

    print(f"Batch size: {batch_size}")
    print(f"Time steps: {time_steps}")
    print(f"\nSample 0 events: {len(events_batch[0])}")
    for event in events_batch[0]:
        print(
            f"  Class {event.class_id}: [{event.start_norm:.3f}, {event.end_norm:.3f}]"
        )

    print(f"\nSample 1 events: {len(events_batch[1])}")
    for event in events_batch[1]:
        print(
            f"  Class {event.class_id}: [{event.start_norm:.3f}, {event.end_norm:.3f}]"
        )

    assert len(events_batch[0]) == 2, "Expected 2 events in sample 0"
    assert len(events_batch[1]) == 1, "Expected 1 event in sample 1"
    print("\n✓ Event target conversion test PASSED")


def test_hungarian_matcher():
    """Test Hungarian matching."""
    print("\n" + "=" * 80)
    print("TEST 2: Hungarian Matching")
    print("=" * 80)

    # Create dummy predictions and targets
    batch_size = 1
    num_queries = 5
    num_classes = 6
    device = "cpu"

    predictions = {
        "class_logits": torch.randn(batch_size, num_queries, num_classes),
        "center": torch.rand(batch_size, num_queries, 1),
        "start": torch.rand(batch_size, num_queries, 1),
        "end": torch.rand(batch_size, num_queries, 1) + 0.1,  # Ensure end > start
        "confidence": torch.randn(batch_size, num_queries, 1),
    }

    # Create dummy targets
    from utils.event_targets import EventInterval

    targets = [
        [
            EventInterval(
                class_id=1,
                start_norm=0.1,
                end_norm=0.3,
                center_norm=0.2,
                start_frame=100,
                end_frame=300,
            ),
            EventInterval(
                class_id=2,
                start_norm=0.5,
                end_norm=0.7,
                center_norm=0.6,
                start_frame=500,
                end_frame=700,
            ),
        ]
    ]

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0)
    matches = matcher(predictions, targets, num_classes=num_classes)

    print(f"Predictions: {num_queries} queries")
    print(f"Targets: {len(targets[0])} events")
    print(f"\nMatching result:")
    print(f"  Matched predictions: {matches[0].pred_indices}")
    print(f"  Matched targets: {matches[0].target_indices}")
    print(f"  Unmatched predictions: {matches[0].unmatched_pred}")
    print(f"  Unmatched targets: {matches[0].unmatched_target}")

    assert len(matches) == 1, "Should have 1 match result for batch size 1"
    print("\n✓ Hungarian matching test PASSED")


def test_detr_loss():
    """Test DETR loss computation."""
    print("\n" + "=" * 80)
    print("TEST 3: DETR Loss Computation")
    print("=" * 80)

    batch_size = 2
    num_queries = 5
    num_classes = 6
    device = "cpu"

    predictions = {
        "class_logits": torch.randn(batch_size, num_queries, num_classes),
        "center": torch.rand(batch_size, num_queries, 1),
        "start": torch.rand(batch_size, num_queries, 1),
        "end": torch.rand(batch_size, num_queries, 1) + 0.1,
        "confidence": torch.randn(batch_size, num_queries, 1),
    }

    from utils.event_targets import EventInterval

    targets = [
        [
            EventInterval(
                class_id=1,
                start_norm=0.1,
                end_norm=0.3,
                center_norm=0.2,
                start_frame=100,
                end_frame=300,
            ),
        ],
        [
            EventInterval(
                class_id=2,
                start_norm=0.4,
                end_norm=0.6,
                center_norm=0.5,
                start_frame=400,
                end_frame=600,
            ),
        ],
    ]

    loss_fn = DETREventLoss(num_classes=num_classes)
    loss_dict = loss_fn(predictions, targets)

    print(f"Batch size: {batch_size}")
    print(f"Queries: {num_queries}")
    print(f"Targets: {sum(len(t) for t in targets)}")
    print(f"\nLoss breakdown:")
    print(f"  Total loss: {loss_dict['loss_total']:.6f}")
    print(f"  Class loss: {loss_dict['loss_class']:.6f}")
    print(f"  BBox loss: {loss_dict['loss_bbox']:.6f}")
    print(f"  Confidence loss: {loss_dict['loss_conf']:.6f}")
    print(f"  Matched: {loss_dict['metrics']['num_matched']}")
    print(f"  Targets: {loss_dict['metrics']['num_targets']}")

    assert isinstance(loss_dict["loss_total"], torch.Tensor), "Loss should be tensor"
    assert loss_dict["loss_total"].item() > 0, "Loss should be positive"
    print("\n✓ DETR loss test PASSED")


def test_metrics():
    """Test event detection metrics."""
    print("\n" + "=" * 80)
    print("TEST 4: Event Detection Metrics")
    print("=" * 80)

    # Create predictions and targets
    batch_size = 1
    num_queries = 5
    num_classes = 6

    predictions_np = {
        "class_logits": np.random.randn(batch_size, num_queries, num_classes).astype(
            np.float32
        ),
        "center": np.random.rand(batch_size, num_queries, 1).astype(np.float32),
        "start": np.random.rand(batch_size, num_queries, 1).astype(np.float32),
        "end": (np.random.rand(batch_size, num_queries, 1) + 0.1).astype(np.float32),
        "confidence": np.random.randn(batch_size, num_queries, 1).astype(np.float32),
    }

    from utils.event_targets import EventInterval

    targets = [
        [
            EventInterval(
                class_id=1,
                start_norm=0.1,
                end_norm=0.3,
                center_norm=0.2,
                start_frame=100,
                end_frame=300,
            ),
            EventInterval(
                class_id=2,
                start_norm=0.5,
                end_norm=0.7,
                center_norm=0.6,
                start_frame=500,
                end_frame=700,
            ),
        ]
    ]

    metrics_fn = EventDetectionMetrics(iou_thresholds=[0.3, 0.5, 0.7])
    metrics = metrics_fn.evaluate_batch(
        predictions_np, targets, confidence_threshold=0.5
    )

    print(f"Predictions: {num_queries} queries")
    print(f"Targets: {len(targets[0])} events")
    print(f"\nMetrics:")
    for key, value in sorted(metrics.items()):
        if "mAP" in key or "AP_class" in key:
            print(f"  {key}: {value:.4f}")

    assert "mAP" in metrics, "Should have mAP metric"
    assert 0 <= metrics["mAP"] <= 1, "mAP should be in [0, 1]"
    print("\n✓ Metrics test PASSED")


def test_trainer():
    """Test trainer instantiation and basic functionality."""
    print("\n" + "=" * 80)
    print("TEST 5: Trainer Instantiation")
    print("=" * 80)

    # Create model
    config = {
        "in_channels": 8,
        "num_classes": 6,
        "depth": 3,
        "kernel_size": 63,
        "stride": [2, 2],
        "dilation": [1, 1, 1],
        "filters_root": 8,
        "norm": "std",
        "feature_dropout": 0.0,
        "bottleneck_attention": False,
        "bottleneck_attn_heads": 2,
        "bottleneck_attn_dropout": 0.0,
        "bottleneck_attn_ff_mult": 2,
        "station_attn_heads": 2,
        "station_attn_dropout": 0.0,
        "station_attn_ff_mult": 2,
        "volcano_name": "NVCHVC",
        "use_distance_attn_bias": False,
        "use_distance_bottleneck_emb": False,
        "use_station_weighted_skips": False,
        "num_queries": 3,
        "query_dim": 64,
        "hidden_dim": 128,
        "num_decoder_heads": 2,
        "num_decoder_layers": 2,
        "decoder_dropout": 0.1,
    }

    device = "cpu"
    dtype = torch.float32

    model = MuSSED(**config)
    trainer = DETRTrainer(model, device=device, dtype=dtype)

    print(f"Model created successfully")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    # Setup optimizer
    trainer.setup_optimizer(learning_rate=1e-4, warmup_steps=100)
    print(f"Optimizer: {trainer.optimizer.__class__.__name__}")
    print(f"Scheduler: {trainer.scheduler.__class__.__name__}")

    # Test forward pass
    batch_x = torch.randn(2, 3, 2048, device=device, dtype=dtype)
    with torch.no_grad():
        output = model(batch_x)

    print(f"\nForward pass:")
    print(f"  Input shape: {tuple(batch_x.shape)}")
    print(f"  Output keys: {list(output.keys())}")
    print(f"  class_logits shape: {output['class_logits'].shape}")
    print(f"  center shape: {output['center'].shape}")
    print(f"  confidence shape: {output['confidence'].shape}")

    assert output["class_logits"].shape == (2, 3, 6), "Unexpected class_logits shape"
    print("\n✓ Trainer test PASSED")


def main():
    """Run all integration tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "MuSSED PHASES 2 & 3 INTEGRATION TEST".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")

    try:
        test_event_targets()
        test_hungarian_matcher()
        test_detr_loss()
        test_metrics()
        test_trainer()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓".center(80))
        print("=" * 80)
        print("\nIntegration pipeline verified successfully!")
        print("Ready for training with real data.\n")

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED ✗".center(80))
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
