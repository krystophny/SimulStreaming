# Whisper Modifications in SimulStreaming

This document explains the modifications made to OpenAI Whisper for SimulStreaming's AlignAtt simultaneous policy.

## Base Version

SimulStreaming is based on **OpenAI Whisper v20230918** (September 18, 2023) with significant modifications to support streaming and simultaneous translation.

## Why Modifications Are Needed

AlignAtt simultaneous policy requires:
- **Incremental decoding**: Process audio chunks as they arrive
- **Efficient KV caching**: Reuse computed key/value tensors across chunks
- **Attention monitoring**: Track encoder-decoder attention for segmentation decisions
- **Streaming-aware initialization**: Control start tokens for continuation

These features are NOT in upstream OpenAI Whisper, which is designed for batch processing of complete audio files.

## Modified Files

### 1. `model.py` (~215 lines changed) ⚠️ HEAVY MODIFICATIONS

**Purpose:** KV cache infrastructure for efficient streaming

#### Cache ID System
Added unique identifiers to track cached tensors across attention layers:

```python
# BEFORE (upstream):
class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)

# AFTER (SimulStreaming):
class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, cache_id: str):
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.key.cache_id = f"{cache_id}_key"
        self.value = nn.Linear(n_state, n_state)
        self.value.cache_id = f"{cache_id}_value"
        self.cache_id = cache_id
```

**Cache naming scheme:**
- Encoder layers: `enc_layer0`, `enc_layer1`, ..., `enc_layer31`
- Decoder self-attention: `dec_layer0_self_attn`, `dec_layer1_self_attn`, ...
- Decoder cross-attention: `dec_layer0_cross_attn`, `dec_layer1_cross_attn`, ...
- Keys/values: `dec_layer0_self_attn_key`, `dec_layer0_self_attn_value`, ...

#### KV Cache Lookup
Changed cache access to use unique IDs:

```python
# BEFORE:
if kv_cache is None or xa is None or self.key not in kv_cache:
    k = self.key(x if xa is None else xa)
    v = self.value(x if xa is None else xa)
else:
    k = kv_cache[self.key]
    v = kv_cache[self.value]

# AFTER:
if kv_cache is None or xa is None or self.key.cache_id not in kv_cache:
    k = self.key(x if xa is None else xa)
    v = self.value(x if xa is None else xa)
else:
    k = kv_cache[self.key.cache_id]
    v = kv_cache[self.value.cache_id]
```

**Why this matters:** Allows `simul_whisper.py` to intercept and manage KV cache via hooks, enabling incremental decoding across streaming chunks.

#### Simplified Type Handling
Removed custom dtype conversion layers:

```python
# BEFORE: Custom layers for dtype conversion
class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), ...)

# AFTER: Use standard PyTorch layers
# (commented out custom classes, use nn.LayerNorm, nn.Linear directly)
```

**Rationale:** Relies on model-wide `.half()` for fp16 instead of per-layer conversions. Simplifies code and improves compatibility.

#### Disabled Scaled Dot Product Attention (SDPA)
```python
class MultiHeadAttention(nn.Module):
    use_sdpa = False  # disabling: https://github.com/linto-ai/whisper-timestamped/issues/212
```

**Reason:** SDPA interferes with attention hooks needed for AlignAtt policy monitoring.

---

### 2. `decoding.py` (~30 lines changed)

**Purpose:** Streaming-aware decoding options

#### Added `add_sot` Option
New parameter to control start-of-transcript token:

```python
@dataclass
class DecodingOptions:
    # ... existing options ...

    # streaming
    add_sot: Optional[bool] = True
```

**Use case:** When continuing from a previous chunk, may not need SOT token.

#### Reordered fp16 Handling
```python
# BEFORE:
def __init__(self, model: "Whisper", options: DecodingOptions):
    self.model = model
    self.options: DecodingOptions = self._verify_options(options)

# AFTER:
def __init__(self, model: "Whisper", options: DecodingOptions):
    self.options: DecodingOptions = self._verify_options(options)
    if self.options.fp16:
        self.model = model.half()
    else:
        self.model = model
```

**Reason:** Apply fp16 conversion based on options before using the model.

#### Bug Fix
```python
# BEFORE:
return TypeError(f"unsupported task: {task}")

# AFTER:
raise TypeError(f"unsupported task: {task}")
```

#### Debug Comments
Added commented-out print statements and Chinese comments (中文注释) for debugging token sequences and decoding state.

---

### 3. `timing.py` (~26 lines changed)

**Purpose:** Debug output for attention-based alignment

#### Chinese Comments
Added explanatory comments about DTW algorithm:

```python
@numba.jit(nopython=True)
def backtrace(trace: np.ndarray):
    i = trace.shape[0] - 1 # trace: (N+1, M+1), i=N
    j = trace.shape[1] - 1 # j=M
    # 边界点其实无意义？ (boundary points are meaningless?)
```

```python
cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf # cost: x[0,0]到x[i-1,j-1]的最小代价
trace = -np.ones((N + 1, M + 1), dtype=np.float32) # trace:
```

#### Debug Print Statements
Added prints to inspect timing alignment:

```python
print("attention", matrix.shape, matrix[:5, :5])
print("num_frames", num_frames)
print("text_indices", text_indices)
print("time", time_indices)
print("text_tokens", text_tokens, tokenizer.decode(text_tokens), len(text_tokens))
print("eot", tokenizer.eot)
```

**Note:** These prints are active (not commented out) and will output to stderr during word timestamp generation.

---

### 4. `triton_ops.py` (NO CHANGES)

Identical to upstream Whisper v20230918.

**Current issue:** Triton 3.x breaks API compatibility (see main README).

---

### 5. `trans_nopad.py` (SimulStreaming-specific)

**NOT in upstream OpenAI Whisper.**

This file appears to be an alternative transcription function. It imports from `whisper.*` modules (not `.whisper`), suggesting it may be unused or from an earlier version.

**Status:** Currently NOT imported or used in SimulStreaming codebase.

---

## Why We Cannot Use `openai-whisper` Package

SimulStreaming **cannot** simply depend on the `openai-whisper` PyPI package because:

1. **KV cache ID system is essential** for AlignAtt:
   - Upstream Whisper has no cache IDs
   - AlignAtt's `simul_whisper.py` expects `cache_id` attributes on layers
   - Hooks in `simul_whisper.py` use `module.cache_id` to manage cache

2. **Streaming options not in upstream**:
   - `add_sot` option doesn't exist in upstream `DecodingOptions`
   - Would break streaming continuation logic

3. **Version pinning issues**:
   - Based on v20230918 (Sept 2023)
   - Upstream has evolved significantly since then
   - Newer versions may have incompatible changes

## Architecture: How KV Cache Works

```
┌─────────────────────────────────────────────────────────┐
│ simul_whisper.py (AlignAtt Policy)                      │
│                                                          │
│  self.kv_cache = {}                                     │
│                                                          │
│  def kv_hook(module, _, net_output):                    │
│      cache_id = module.cache_id  ◄── Needs this!       │
│      if cache_id not in self.kv_cache:                  │
│          self.kv_cache[cache_id] = net_output           │
│      else:                                              │
│          self.kv_cache[cache_id] = torch.cat(...)       │
│                                                          │
│  for block in model.decoder.blocks:                     │
│      block.attn.key.register_forward_hook(kv_hook)      │
│      block.attn.value.register_forward_hook(kv_hook)    │
│      block.cross_attn.key.register_forward_hook(...)    │
│      block.cross_attn.value.register_forward_hook(...)  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Modified Whisper model.py                               │
│                                                          │
│  class MultiHeadAttention:                              │
│      def __init__(self, ..., cache_id: str):            │
│          self.key.cache_id = f"{cache_id}_key"          │
│          self.value.cache_id = f"{cache_id}_value"      │
│                                                          │
│      def forward(self, ..., kv_cache):                  │
│          if self.key.cache_id not in kv_cache:          │
│              k = self.key(...)  # ◄── Hook captures     │
│              v = self.value(...)                        │
│          else:                                          │
│              k = kv_cache[self.key.cache_id]  # Reuse!  │
│              v = kv_cache[self.value.cache_id]          │
└─────────────────────────────────────────────────────────┘
```

**Flow:**
1. `simul_whisper.py` registers hooks on key/value layers
2. Hooks capture outputs and store in `self.kv_cache[module.cache_id]`
3. Modified Whisper checks `kv_cache[self.key.cache_id]` to reuse tensors
4. Enables incremental decoding: only compute new tokens, reuse past K/V

## Maintenance Strategy

**Current approach:** Vendor modified Whisper code in `simul_whisper/whisper/`

**Pros:**
- ✅ Full control over modifications
- ✅ No dependency on upstream changes
- ✅ Can apply fixes (e.g., Triton 3.x compatibility)

**Cons:**
- ❌ Must manually port upstream bug fixes
- ❌ Miss out on upstream improvements
- ❌ Increased maintenance burden

**Alternatives considered:**
1. **Monkey-patch upstream**: Too fragile, would break across versions
2. **Subclass upstream**: Doesn't work for dataclasses and module structure
3. **Fork openai-whisper**: Creates separate package, harder to distribute

**Recommendation:** Keep vendored code, document modifications (this file), and selectively port critical upstream fixes.

## Triton Version Constraint

The vendored Whisper code requires `triton>=2.0.0,<3` because:
- Triton 3.0 broke the `kernel.src` modification API used in `triton_ops.py`
- Upstream OpenAI Whisper has fixed this for Triton 3.x
- We should port this fix: see https://github.com/openai/whisper/discussions/2597

## Summary

| File | Lines Changed | Severity | Purpose |
|------|--------------|----------|---------|
| `model.py` | ~215 | ⚠️ HEAVY | KV cache infrastructure with unique IDs |
| `decoding.py` | ~30 | MODERATE | Streaming options (`add_sot`), fp16 handling |
| `timing.py` | ~26 | LIGHT | Debug prints and Chinese comments |
| `triton_ops.py` | 0 | NONE | Identical to v20230918 (needs Triton 3.x fix) |
| `trans_nopad.py` | N/A | UNUSED | SimulWhisper-specific (not imported) |

**Critical dependencies:**
- `cache_id` system in `model.py` is **essential** for AlignAtt
- Cannot replace with upstream without breaking functionality
- Must maintain vendored fork with modifications
