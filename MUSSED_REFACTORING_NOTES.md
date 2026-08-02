# MuSSED Refactoring Summary

## Changes Made

### 1. MuSSED.py Refactoring

#### Renamed Classes
- **MuSSEDEncoder** → **TemporalEncoder**
  - Now presented as a standalone temporal encoder (not MuSSeg-specific)
  - Internally uses MuSSeg architecture but abstracted away in API
  - Public interface: `in_channels`, `num_classes`, encoder architecture params

#### Consolidated Parameters
- **Removed:** `encoder_classes` parameter
- **Kept:** Single `num_classes` parameter (applies to both encoder and detection head)
- **Reason:** These should always be the same - both encoder and detector operate on same event class set

#### Removed Parameters
- **`pre_bottleneck_station_attn_merge`** ❌ REMOVED
  - This was an internal detail that added unnecessary complexity
  - Functionality: Used attention-based merging of stations before bottleneck attention
  - Impact: Simplifies station merging logic

- **`station_message_*` parameters** ❌ REMOVED
  - `station_message_levels`
  - `station_message_aggregation`
  - `station_message_ratio`
  - **What they did:** Permutation-invariant message passing between stations at specified encoder levels (pair-wise interactions, learned weighted aggregation, etc.)
  - **Why removed:** Not needed for v1; kept encoder simpler; can be added back if needed
  - **Kept:** `station_attention_levels` for global station attention (different mechanism)

#### Clarified Station Interaction
**Important:** `station_interaction` and `station_attention_levels` are **mutually exclusive alternatives**:

```python
# Option 1: NO station interaction
station_interaction = "none"
station_attention_levels = []

# Option 2: Message passing between stations
station_interaction = "late_station_message"
# (station_attention_levels will be automatically overridden)

# Option 3: Global cross-attention over stations
station_interaction = "late_attention"
station_attention_levels = [3]  # e.g., apply at encoder level 3
```

For the test script, we use **Option 3** (late_attention):
```python
"station_interaction": "late_attention",
"station_attention_levels": [3],  # For depth=4, final level is 3 (0-indexed)
```

### 2. Test Script Updates (02c_test_mussed.py)

#### Updated Configuration
Now uses the **best performing ablation** from model registry:
```
musseg_d4_r16_s222_k127:
- depth: 4
- filters_root: 16
- stride: [2, 2, 2]
- kernel_size: 127
- dilation: [1, 1, 1, 1]
- bottleneck_attention: True
- station_interaction: late_attention
```

#### Parameter Fixes
- Changed `in_channels: 3` → `in_channels: 8` (matches NVCHVC data format)
- Changed `stride: 4` → `stride: [2, 2, 2]` (per-level strides)
- Changed `station_attention_levels: [4]` → `station_attention_levels: [3]` 
  - **Why:** depth=4 means encoder levels are 0, 1, 2, 3 (0-indexed)
  - Late attention should be at level 3, not 4
- Removed `station_message_levels`, `station_message_aggregation`, `station_message_ratio`
- Removed `pre_bottleneck_station_attn_merge`
- Removed `encoder_classes` (redundant with `num_classes`)

---

## API Comparison

### Before
```python
model = MuSSED(
    in_channels=3,
    encoder_classes=6,      # ← Redundant
    num_classes=6,          # ← Redundant
    pre_bottleneck_station_attn_merge=False,  # ← Removed
    station_message_levels=[],  # ← Removed
    station_message_aggregation="sum",  # ← Removed
    station_message_ratio=1.0,  # ← Removed
    station_attention_levels=[4],  # ← Fixed indexing
    ...
)
```

### After
```python
model = MuSSED(
    in_channels=8,
    num_classes=6,          # ← Single class parameter
    station_attention_levels=[3],  # ← Correct for depth=4
    ...
    # (pre_bottleneck_station_attn_merge, station_message_* all removed)
)
```

---

## What These Parameters Do (Reference)

### Kept: `station_interaction`
- **`"none"`**: No station-level interaction, stations processed independently
- **`"late_station_message"`**: Permutation-invariant pairwise message passing (pair interactions)
- **`"late_attention"`**: Global cross-attention over stations (each station attends all others)

### Kept: `station_attention_levels`
- List of encoder depth levels where **global station attention** is applied
- Only meaningful if `station_interaction="late_attention"` or explicitly set
- For `depth=4`: valid levels are `[0, 1, 2, 3]` (0-indexed)
- Example: `[3]` means apply station attention only at the bottleneck (finest features)

### Removed: `station_message_*`
- **Message Passing:** Different from attention
  - Permutation equivariant (invariant to station ordering)
  - Pairwise interactions: station i processes messages from each other station j
  - Learned aggregation (sum vs attention-weighted)
- **Why removed for v1:**
  - Added complexity without clear benefit for initial testing
  - `late_attention` (global attention) is sufficient and cleaner
  - Can reintroduce if ablations show improvement

### Removed: `pre_bottleneck_station_attn_merge`
- **What it did:** Merge stations into single [B, C, T] BEFORE bottleneck attention
- **How:** Use attention (not max) to compute station merge weights
- **Why removed:**
  - Extra option that duplicates functionality of `_merge_stations()`
  - Adds conditional logic without clear justification
  - Keep codebase simpler for v1

---

## Test Configuration Details

```python
MUSSED_CONFIG = {
    # Encoder: Best ablation (musseg_d4_r16_s222_k127)
    "in_channels": 8,           # NVCHVC waveform data
    "num_classes": 6,           # VT, LP, TR, AV, IC, background
    "depth": 4,                 # 4 levels of downsampling
    "kernel_size": 127,         # Large receptive field
    "stride": [2, 2, 2],        # Progressive 2x downsampling
    "filters_root": 16,         # 16, 32, 64, 128, 256 channels per level
    
    # Bottleneck & station interaction
    "bottleneck_attention": True,      # Temporal attention at bottleneck
    "station_interaction": "late_attention",
    "station_attention_levels": [3],   # Apply station attention at level 3
    
    # Detection head
    "num_queries": 3,           # Detect up to 3 events
    "query_dim": 128,           # Query embedding dimension
    "num_decoder_layers": 2,    # Lightweight transformer decoder
    "num_decoder_heads": 4,     # Multi-head attention
}
```

---

## Next Steps

1. ✅ Run `02c_test_mussed.py` to verify model loads and produces valid outputs
2. Review output shapes, value ranges, and gradient flow
3. Move to Phase 2: Implement losses, Hungarian matching, evaluation metrics
4. Move to Phase 3: Build training loop and integrate with existing pipeline
