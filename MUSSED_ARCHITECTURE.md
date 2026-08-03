# MuSSED: Multi-Station Seismic Event Detection

## Overview

**MuSSED** (Multi-Station Seismic Event Detection) is a DETR-inspired (Detection Transformer) deep learning model designed to detect seismic events in multi-station waveform data. Unlike traditional per-timestep segmentation approaches, MuSSED uses a query-based detection paradigm to identify and localize a **fixed set of events** (by default 3) from continuous waveform streams.

### Key Innovation
- **Encoder-Decoder Architecture**: Uses a U-Net-like temporal encoder with station awareness, followed by a transformer-based detection decoder
- **Multi-Station Awareness**: Incorporates station-level attention mechanisms to leverage inter-station relationships
- **Query-Based Detection**: Learns a fixed set of "event queries" that compete to detect events in the data
- **Event Localization**: Predicts event class, center time, start time, end time, and confidence for each detected event

---

## High-Level Architecture

```
Input: [B, S, T]
   ↓
┌─────────────────────────────────────────┐
│  TEMPORAL ENCODER (TemporalEncoder)      │ ← MuSSeg backbone
│  - Downsampling path (U-Net)             │
│  - Station-aware attention at final level│
│  - Bottleneck temporal attention         │
│  Output: [B, C, T']                      │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  TEMPORAL PROJECTION                     │
│  Projects features: C → query_dim        │
│  Output: [B, T', query_dim]              │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  POSITIONAL ENCODING                     │
│  Adds sinusoidal position encodings      │
│  Output: [B, T', query_dim]              │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  TRANSFORMER DECODER                     │
│  - Queries (learnable): [B, Nq, query_dim]│
│  - Memory (features): [B, T', query_dim] │
│  - Self-attention on queries             │
│  - Cross-attention to temporal features  │
│  Output: [B, Nq, query_dim]              │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  DETECTION HEAD (DetectionHead)          │
│  Parallel prediction branches:           │
│  - class_logits: [B, Nq, num_classes]    │
│  - center: [B, Nq, 1]                    │
│  - start: [B, Nq, 1]                     │
│  - end: [B, Nq, 1]                       │
│  - confidence: [B, Nq, 1]                │
└─────────────────────────────────────────┘
```

---

## Input/Output Specification

### Input Tensor Format

```python
x: torch.Tensor
  Shape: [B, S, T]
  where:
    B = batch size (e.g., 4-32)
    S = number of stations (variable, typically 2-20)
    T = time samples (variable, typically 2048-32768 samples)
    
  Example: [8, 10, 8192] → 8 batches, 10 stations, 8192 time samples
```

### Output Dictionary

```python
predictions: Dict[str, torch.Tensor]
{
    "class_logits": torch.Tensor,        # [B, Nq, num_classes] 
                                         # Class probabilities (softmax applied in loss)
    
    "center": torch.Tensor,              # [B, Nq, 1]
                                         # Normalized center time (∈ [0, 1] or unbounded)
    
    "start": torch.Tensor,               # [B, Nq, 1]
                                         # Normalized start time
    
    "end": torch.Tensor,                 # [B, Nq, 1]
                                         # Normalized end time
    
    "confidence": torch.Tensor,          # [B, Nq, 1]
                                         # Objectness score (sigmoid applied in loss)
    
    "encoder_features": torch.Tensor,    # [B, C, T']
                                         # Bottleneck features for interpretability
}

where:
  Nq = num_queries (default 3)
  num_classes = 6 (VT, LP, TR, AV, IC, background)
  C = encoder output channels
  T' = downsampled time dimension
```

### Example Shapes with Default Config

```
Input:  [8, 10, 8192]  (8 batches, 10 stations, 8192 samples)
                  ↓
Encoder: [8, 64, 256]  (64 channels, 32x downsampling)
                  ↓
Projected: [8, 256, 128]  (256 time steps, 128-dim features)
                  ↓
Decoder Input: queries [8, 3, 128], memory [8, 256, 128]
                  ↓
Decoder Output: [8, 3, 128]  (3 queries, 128-dim embeddings)
                  ↓
Predictions:
  class_logits: [8, 3, 6]       (3 queries, 6 classes)
  center:       [8, 3, 1]       (3 event centers)
  start:        [8, 3, 1]       (3 event starts)
  end:          [8, 3, 1]       (3 event ends)
  confidence:   [8, 3, 1]       (3 objectness scores)
```

---

## Forward Pass Step-by-Step

This section provides a detailed walkthrough of the forward pass in the order data flows through the model.

### Step 1: Input Validation & Encoding

```python
# Input arrives as [B, S, T]
x = input_waveforms  # [B, S, T]

# Temporal Encoder processes multi-station waveforms
encoder_features = self.encoder(x)  # [B, C, T']
```

**What happens inside `TemporalEncoder`:**

1. **Reshape for station processing**: `[B, S, T] → [B, S, 1, T]`
   - Each station is processed independently initially
   
2. **Downsampling path** (CNN encoder):
   - Apply shared convolution: `[B, S, 1, T] → [B, S, F₀, T]`
   - For each of `depth` levels:
     - Apply station-wise convolution (same size)
     - Apply station attention at the final level (depth-1)
     - Downsample: `[B, S, Fᵢ, T] → [B, S, Fᵢ₊₁, T/2ⁱ]`

3. **Bottleneck** (final level):
   - If `bottleneck_attention=True`: Apply temporal attention across time
   - Merge stations: `[B, S, F_bottleneck, T'] → [B, F_bottleneck, T']`

**Output**: `[B, C, T']` where:
- `B` = batch size (unchanged)
- `C` = number of channels at bottleneck (e.g., 64)
- `T'` = downsampled time (e.g., 8192 / 2^4 = 512)

---

### Step 2: Temporal Projection

```python
# Project encoder channels to query dimension
encoder_proj = encoder_features.transpose(1, 2)  # [B, C, T'] → [B, T', C]
encoder_proj = self.temporal_proj(encoder_proj)   # [B, T', C] → [B, T', query_dim]
```

**Purpose**: 
- Reduce or maintain dimensionality for decoder (e.g., 64 → 128)
- Creates a common embedding space for queries and memory
- Linear layer: `nn.Linear(C, query_dim)`

**Output**: `[B, T', query_dim]`

---

### Step 3: Positional Encoding

```python
# Add sinusoidal positional encodings to preserve temporal ordering
pos_enc = self.positional_encoding(encoder_proj)  # [B, T', query_dim]
memory = encoder_proj + pos_enc                   # Element-wise addition
```

**Purpose**:
- Transformer blocks are permutation-invariant by nature
- Positional encoding injects information about **absolute temporal positions**
- Uses sinusoidal functions: 
  - `PE(t, 2i) = sin(t / 10000^(2i/d))`
  - `PE(t, 2i+1) = cos(t / 10000^(2i/d))`

**Why Sinusoidal?**
- Wavelengths increase exponentially (fine → coarse temporal scales)
- Allows the model to learn absolute and relative temporal positions easily
- Low memory overhead (computed on-the-fly)

**Output**: `memory [B, T', query_dim]` with position information

---

### Step 4: Query Expansion

```python
# Expand learnable event queries to batch size
queries = self.event_queries.expand(batch_size, -1, -1)  # [1, Nq, query_dim] → [B, Nq, query_dim]
```

**What are queries?**
- **Learnable Parameters**: `nn.Parameter(torch.randn(1, Nq, query_dim) / sqrt(query_dim))`
- **Initialization**: Normalized Gaussian (std = 1/√query_dim)
- **Count**: Fixed number `Nq` (default 3 for volcano datasets)
- **Interpretation**: Each query is a "detector" looking for one specific event

**Intuition**:
- In DETR, queries learn to attend to different spatial locations
- In MuSSED, queries learn to attend to different temporal event patterns
- All queries can compete to detect different events simultaneously

**Output**: `[B, Nq, query_dim]`

---

### Step 5: Transformer Decoder

```python
# Transformer decoder processes queries in context of temporal features
decoder_out = self.decoder(queries, memory)  # [B, Nq, query_dim]
```

**What the decoder does:**

**Layer 1: Self-Attention (Query-to-Query)**
```
Input: queries [B, Nq, query_dim]
Output: updated_queries [B, Nq, query_dim]

For each query:
  - Compute attention scores to all other queries
  - Aggregate information: "Are other queries detecting events?"
  - Update query embeddings
  
Result: Queries become aware of each other's detections
```

**Layer 2: Cross-Attention (Query-to-Memory)**
```
Query: queries [B, Nq, query_dim]
Key/Value: memory [B, T', query_dim]
Output: updated_queries [B, Nq, query_dim]

For each query:
  - Attend to all time steps in memory
  - Learn "weighted sum" over temporal features
  - Extract relevant temporal patterns
  
Result: Queries extract event-specific features from waveforms
```

**Repeated**: Stack of `num_decoder_layers` (default 2) encoder-decoder blocks

**Output**: `[B, Nq, query_dim]` (refined query embeddings)

---

### Step 6: Detection Head (Prediction Heads)

```python
# Parallel MLPs predict event properties
predictions = self.detection_head(decoder_out)  # Dict[str, Tensor]
```

**Architecture** (for each prediction head):
```
Input: [B, Nq, query_dim]
  ↓
Linear Layer 1: query_dim → hidden_dim (e.g., 128 → 256)
  ↓
ReLU Activation
  ↓
Linear Layer 2: hidden_dim → output_dim (e.g., 256 → num_classes/1)
  ↓
Output: [B, Nq, output_dim]
```

**Five Independent Heads:**

1. **Class Head** → `[B, Nq, 6]`
   - Predicts class probabilities (raw logits, softmax applied later in loss)
   - Output: 6 classes (VT, LP, TR, AV, IC, background)

2. **Center Head** → `[B, Nq, 1]`
   - Predicts normalized event center time
   - Range: [0, 1] or unbounded (depends on loss function)

3. **Start Head** → `[B, Nq, 1]`
   - Predicts normalized event start time

4. **End Head** → `[B, Nq, 1]`
   - Predicts normalized event end time

5. **Confidence Head** → `[B, Nq, 1]`
   - Predicts objectness score (sigmoid applied in loss → [0, 1])
   - Indicates: "Is there an event to detect?"

**Output**: Dictionary with all predictions

---

### Step 7: Return with Encoder Features

```python
predictions["encoder_features"] = encoder_features  # [B, C, T']
return predictions
```

**Why include encoder features?**
- **Interpretability**: Visualize bottleneck activations
- **Analysis**: Understand what the encoder learned
- **Debugging**: Check if encoder is producing meaningful features
- **Optional use**: Could feed back to decoder for iterative refinement

---

## Complete Forward Pass Pseudocode

```python
def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Full forward pass of MuSSED
    
    Input:  x [B, S, T]
    Output: predictions Dict with class_logits, center, start, end, confidence
    """
    
    # STAGE 1: Encode waveforms to temporal features
    # ─────────────────────────────────────────────
    encoder_features = self.encoder(x)              # [B, S, T] → [B, C, T']
    batch_size, _, t_prime = encoder_features.shape
    
    
    # STAGE 2: Project and position-encode features
    # ─────────────────────────────────────────────
    memory = encoder_features.transpose(1, 2)      # [B, C, T'] → [B, T', C]
    memory = self.temporal_proj(memory)            # [B, T', C] → [B, T', query_dim]
    pos_enc = self.positional_encoding(memory)     # Sinusoidal encodings
    memory = memory + pos_enc                       # Add positions [B, T', query_dim]
    
    
    # STAGE 3: Prepare queries
    # ─────────────────────────────────────────────
    queries = self.event_queries.expand(batch_size, -1, -1)  # [1, Nq, d] → [B, Nq, d]
    
    
    # STAGE 4: Decode (Transformer)
    # ─────────────────────────────────────────────
    # queries: [B, Nq, query_dim]
    # memory:  [B, T', query_dim]
    decoder_out = self.decoder(queries, memory)    # [B, Nq, query_dim]
    
    
    # STAGE 5: Predict events
    # ─────────────────────────────────────────────
    predictions = self.detection_head(decoder_out)  # Dict with:
                                                    #   class_logits: [B, Nq, 6]
                                                    #   center:       [B, Nq, 1]
                                                    #   start:        [B, Nq, 1]
                                                    #   end:          [B, Nq, 1]
                                                    #   confidence:   [B, Nq, 1]
    
    
    # STAGE 6: Add encoder features for analysis
    # ─────────────────────────────────────────────
    predictions["encoder_features"] = encoder_features  # [B, C, T']
    
    return predictions
```

---

## Key Components Deep Dive

### 1. TemporalEncoder (The Backbone)

**Based on**: MuSSeg architecture (temporal CNN encoder with station awareness)

**Key Features**:
- **Station-Aware**: Can process 2-20 stations in a single forward pass
- **Late Station Attention**: At the final encoder level (depth-1), stations attend to each other
- **Bottleneck Attention**: Optional temporal attention at the bottleneck
- **Distance Priors**: Optional station distance biases

**Configuration (Best Ablation)**:
```python
depth=4              # 4 downsampling levels → 2^4 = 16x compression
kernel_size=127      # Large kernel for seismic signals
stride=[2,2,2]       # 2x downsampling per level
filters_root=16      # Base filters, doubled each level (16, 32, 64, 128)
```

**Output**: `[B, C, T']` where `C ≈ 64` and `T' ≈ T/16`

---

### 2. PositionalEncoding (Temporal Context)

**Type**: Sinusoidal (from Transformer paper)

**Formula**:
```
PE(t, 2i)   = sin(t / 10000^(2i/d))
PE(t, 2i+1) = cos(t / 10000^(2i/d))

where:
  t = time step (0 to T'-1)
  i = dimension index (0 to d/2-1)
  d = embedding dimension (query_dim)
```

**Properties**:
- Fixed (non-learnable)
- Provides absolute position information
- Wavelengths span multiple orders of magnitude
- Low memory footprint (computed on-the-fly)

---

### 3. DETRTransformerDecoder (Event Detection Engine)

**Architecture**: Standard PyTorch TransformerDecoder

**Per Layer**:
1. **Self-Attention**: Queries attend to other queries
2. **Cross-Attention**: Queries attend to memory (temporal features)
3. **Feed-Forward**: MLP with hidden_dim

**Stacking**: `num_decoder_layers` (default 2) repeated blocks

**Attention Formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d) V

Query-to-Query:
  Q, K, V = all from decoder_out

Query-to-Memory:
  Q = decoder_out
  K, V = memory (encoder features)
```

---

### 4. DetectionHead (Event Properties)

**5 Parallel MLPs**:

```python
class_head:
  Linear(query_dim, hidden_dim)
  → ReLU
  → Linear(hidden_dim, num_classes)   # Output logits (softmax later)

center_head:
  Linear(query_dim, hidden_dim)
  → ReLU
  → Linear(hidden_dim, 1)              # Output single regression value

start_head, end_head, confidence_head:
  # Similar structure
```

**Design Rationale**:
- **Independent heads**: Each predicts different aspects
- **Shared query representation**: All benefit from transformer decoding
- **Shallow MLPs**: 2 layers sufficient for low-dim → prediction
- **No activation on output**: Applied during loss computation (softmax, sigmoid)

---

## Typical Training Setup

### Loss Function (Conceptual)

```python
total_loss = (
    classification_loss +           # CrossEntropyLoss(class_logits, labels)
    localization_loss +             # SmoothL1Loss(center/start/end, targets)
    confidence_loss                 # BCEWithLogitsLoss(confidence, is_event)
)
```

### Hungarian Matching (DETR-style)

- Queries need to be **assigned to ground-truth events**
- Hungarian algorithm finds optimal matching
- Cost: `(1 - P(class)) + λ_coord * |predictions - targets|`

### Data Format for Training

```python
batch = {
    "waveforms": torch.Tensor,     # [B, S, T]
    "events": list[Dict],           # Per-sample event annotations
}

event = {
    "class": int,                   # 0-5
    "start": float,                 # Normalized [0, 1]
    "center": float,                # Normalized [0, 1]
    "end": float,                   # Normalized [0, 1]
}
```

---

## Inference (Testing)

### Example Usage

```python
import torch
from models.MuSSED import MuSSED

# Initialize model
model = MuSSED(
    in_channels=8,
    num_classes=6,
    num_queries=3,
    query_dim=128,
    depth=4,
    kernel_size=127,
)
model.eval()

# Forward pass
with torch.no_grad():
    waveforms = torch.randn(1, 10, 8192)  # [batch=1, stations=10, time=8192]
    predictions = model(waveforms)

# Process predictions
class_logits = predictions["class_logits"]      # [1, 3, 6]
center_pred = predictions["center"]              # [1, 3, 1]
confidence_pred = predictions["confidence"]      # [1, 3, 1]

# Post-processing
class_pred = torch.argmax(class_logits, dim=-1)  # [1, 3]
confidence = torch.sigmoid(confidence_pred)      # [1, 3, 1] ∈ [0, 1]

# Filter by confidence threshold
threshold = 0.5
for i in range(3):  # For each query
    if confidence[0, i, 0] > threshold:
        event_class = class_pred[0, i].item()
        event_center = center_pred[0, i, 0].item()
        print(f"Detected: Class {event_class} at normalized time {event_center:.3f}")
```

---

## Hyperparameter Interpretation

| Parameter | Default | Role |
|-----------|---------|------|
| `num_queries` | 3 | Max simultaneous detectable events |
| `query_dim` | 128 | Embedding dimension (higher = more expressive) |
| `hidden_dim` | 256 | Intermediate dimension in detection heads |
| `depth` | 4 | Encoder downsampling levels (4 = 16x compression) |
| `kernel_size` | 127 | Receptive field for convolutions |
| `bottleneck_attention` | True | Temporal attention at deepest layer |
| `station_attn_heads` | 4 | Multi-head attention splits |
| `num_decoder_layers` | 2 | Transformer decoder depth |

---

## Computational Complexity

### Memory Scaling

```
Encoder: O(B * S * T * max_channels)
  where max_channels ≈ filters_root * 2^depth ≈ 128 for depth=4

Decoder: O(B * Nq * T' * query_dim)
  where T' = T / 2^depth ≈ T/16

Attention: O(B * T'^2 * query_dim)  or  O(B * Nq^2 * query_dim)
  Cross-attention is quadratic in memory time, not queries
```

### Time Complexity (FLOPs)

- Encoder downsampling: Linear in input size
- Decoder cross-attention: ~4x more FLOPs than encoder
- Detection heads: Negligible (shallow MLPs)

**Typical latency**: ~50-200ms per batch on single GPU (V100/A100)

---

## Key Takeaways

1. **Query-Based Approach**: Fixed number of "detectors" compete to find events
2. **Transformer Decoder**: Powerful self/cross-attention mechanisms for event reasoning
3. **Temporal Encoding**: Absolute positions preserved via sinusoidal encodings
4. **Station Awareness**: U-Net encoder handles multi-station relationships
5. **Parallel Predictions**: Center/start/end/confidence predicted independently per query
6. **End-to-End Trainable**: No hand-crafted features; all learned from data

---

## References & Inspiration

- **DETR** (Carion et al., 2020): Original Detection Transformer for objects
- **MuSSeg**: Underlying U-Net encoder with station interaction
- **Transformer Architecture** (Vaswani et al., 2017): Self-attention mechanisms
