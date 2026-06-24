**4.- correr cross volcano finetuning para ablations**
**5.- correr cross volcano finetuning para UNets**
**6.- nuevo seg-to-target for object detection class**
dataset class must now:

For detection, you can simply add a conversion step:

events = masks_to_events(Y)

where:

events = [
    (start, end, class_id),
    (start, end, class_id),
    ...
]

This operation is trivial computationally. You're just finding contiguous runs of 1s.
**7.- nuevo modelo para obj detection:** (preguntar a copilot cómo implementarlo mejor, )
Architecture:

Keep encoder, GraphSAGE layers, station attention, and bottleneck self-attention.
Remove decoder and segmentation head.
Attach a detection head at the bottleneck.
Detection head outputs per temporal cell:
objectness,
center_offset,
duration,
class_logits[6].

Target assignment:

Downsample temporal dimension according to bottleneck resolution.
Assign each event to the cell containing its center.
Train responsible cell with:
objectness=1,
center_offset,
duration,
class label.
All other cells:
objectness=0.

Loss:

BCE loss for objectness.
L1 loss for center offset.
L1 loss for duration.
Cross-entropy for class prediction.

Inference:

Apply confidence threshold.
Convert center+duration to start/end.
Perform 1D non-maximum suppression using temporal IoU.

Evaluation:

Event Precision.
Event Recall.
Event F1.
mAP@0.5 temporal IoU.
Mean start error.
Mean end error.

Implement the detector so it supports multiple events within the same 8192-sample window.
