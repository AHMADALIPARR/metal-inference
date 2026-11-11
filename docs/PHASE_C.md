# Phase C — Complete Vulkan compute-kernel implementation

Branch: `feat/vulkan-int4-phase-c` (from Phase B `f4005d4`). Local only.
**Do not push.** No GPU execution exists yet: this machine has a
Vulkan loader (1.3.275) and no ICD, so every dispatch is
BLOCKED_NO_PHYSICAL_DEVICE. A constructed command graph is not an
executed inference pass.

## 1. The three INT4 contracts — resolved, not unified

All three source formats are preserved as distinct, named formats.
Nothing is reinterpreted through an implicit default.

| Format | Source | Nibble | Scale | Group | Tensor descriptor value |
|---|---|---|---|---|---|
| A `A_SignedPerRow` | `Transformer.metal` `dequant_i4` + `types.h` | signed, −8..7 | per-row `half*` | none (0) | `QuantFormat::A_SignedPerRow` |
| B `B_Offset75PerGroup32` | `int4_transformer.metal` `dequantNibble` | unsigned, w = (nib − 7.5) × scale | per-group `half2`, low half read | 32 | `QuantFormat::B_Offset75PerGroup32` |
| C `C_SignedPerGroup64` | `fused_qkv_tiled.metal` `dequant_int4` | signed, −8..7 | per-group flat `half*` | 64 | `QuantFormat::C_SignedPerGroup64` |

Nibble order agrees everywhere: even index = low nibble, odd = high
nibble, 2 weights per byte. (`Vulkan/include/tensor_abi.h`,
`Vulkan/include/int4_abi.h`.)

- Every tensor carries an explicit `quantization_format` in its
  `TensorDescriptor` (`{dtype, shape, strides, quantization_format,
  quantization_group_size, scale_dtype, scale_layout, buffer_offset,
  alignment}`), validated before dispatch by `validate_descriptor`:
  rank, shape, strides, alignment, allocation bounds, format/group
  consistency, and divisibility rules (A: in_dim even; B: in_dim %
  32 == 0; C: in_dim % 64 == 0 — see §4).
- `check_format_compatible` refuses reinterpretation: a kernel
  expecting one format will not accept a tensor declared as another.
- Quantized rank-2 shape is in **logical lanes** [rows, in_dim];
  storage is packed: `data_bytes() = rows * ceil(in_dim / 2)`;
  strides = [packed_row_bytes, 1].
- Scale bytes: A = rows × 2, B = rows × groups × 4 (half2),
  C = rows × groups × 2.
- `matvec_fmt` scale-pointer contract: one flat
  `const uint16_t* scales`. A reads `scales[row]`; B reinterprets
  as uint32 half2 slots and reads the low half of slot
  `row*groups + g`; C reads `scales[row*groups64 + g]`.
- CPU references: `dequant_a_at`, `dequant_c_at`, `quantize_row_a`,
  `quantize_row_c`, `matvec_fmt` (all three formats), verified
  against double-precision references and exhaustive bitwise
  roundtrips (all 16 nibble values per format; A/C exhaustive).

## 2. Per-kernel table

All 27 shaders compile with glslangValidator to SPIR-V (Vulkan
1.2), pass `spirv-val`, and pass the table-driven reflection check
(bindings set 0 sequential, push-constant size, spec constants).
Each has an independent CPU reference in
`Vulkan/src/kernel_ref.cpp`, verified against hand-calculable cases.

| Shader | Metal source | ABI (bindings / push bytes) | CPU ref check |
|---|---|---|---|
| `embedding` (B) | `Transformer.metal` | 3 / 4 | bitwise golden (B) |
| `int4_matvec` (B) | `Transformer.metal` | 4 / 8 | golden, 5e-7 (B) |
| `rmsnorm` | `normalization.metal` rmsnorm/final_norm | 3 / 8 | hand case + hidden=3072, 1e-5 |
| `rmsnorm_f32` | float-activation variant | 3 / 4 | hand case, 1e-5 |
| `layernorm` | `normalization.metal` layer_norm | 4 / 8 | hand case, 1e-5 |
| `rope` | `rope.metal` rope_qk (interleaved pairs) | 2 / 16 | hand cases, K-once check |
| `rope_split` | `gqa_strided.metal` rope_apply (half-split) | 2 / 24 | hand case, 1e-6 |
| `fused_qkv` | `fused_qkv_tiled.metal` (Scheme C) | 10 / 16 | composed via decode_layer |
| `int4_matvec_fmt` | A/B/C matvec (one shader, format in push) | 4 / 12 | all 3 formats vs double refs |
| `kv_store` | `kv_cache.metal` kv_store | 4 / 20 | roundtrip, layer aliasing |
| `kv_fetch` | `kv_cache.metal` kv_fetch | 4 / 20 | roundtrip |
| `kv_clear` | `int4_transformer.metal` secureClearKV | 1 / 4 | zeroing |
| `gqa_attention` | `gqa_strided.metal` gqa_attention (two-pass) | 4 / 28 | hand case, 1e-6 |
| `gqa_online` | `int4_transformer.metal` gqaAttention (online softmax) | 4 / 20 | agrees with strided, 1e-6 |
| `attention_scores` | `attention.metal` attention_scores | 3 / 16 | hand case, 1e-6 |
| `softmax` | `attention.metal` stable_softmax | 1 / 12 | rows sum to 1 |
| `attention_value` | `attention.metal` attention_value | 3 / 20 | == fused gqa, 1e-6 |
| `swiglu` | `mlp.metal` swiglu (Scheme A, clamped silu) | 6 / 12 | hand + full, 1e-5 |
| `swiglu_act` | `int4_transformer.metal` swigluFFN (unclamped silu) | 3 / 4 | hand case |
| `mlp_down` | `mlp.metal` mlp_down (Scheme A) | 4 / 12 | full, 1e-5 |
| `residual` | `normalization.metal` residual_and_norm (residual half) | 3 / 8 | bitwise |
| `lora_delta` | `lora_delta.metal` apply_lora_delta | 4 / 16 | hand case; rank ≤ 64 enforced |
| `lora_down` | `int4_transformer.metal` loraDown | 3 / 8 | hand case |
| `lora_up` | `int4_transformer.metal` loraUp | 3 / 12 | hand case |
| `logits` | `output.metal` logits_i4 (Scheme A) | 4 / 12 | full, 1e-5 |
| `argmax` | `output.metal` token_select (first max wins) | 2 / 4 | ties |
| `temperature_softmax` | `output.metal` temperature_softmax | 1 / 8 | sums to 1 |

FP16 in GLSL moves as uint bit patterns (Phase B convention);
each shader carries its own half↔float RNE helpers. `length` is a
reserved word in GLSL: the `length` push-constant member in
`attention_scores`, `attention_value` and `softmax` is spelled
`len`.

## 3. Command graph (forward pass, all layers)

`Vulkan/include/command_graph.h`, `Vulkan/src/command_graph.cpp`.
`build_forward_graph(ModelDims, layers, seq_len, out, why)`:

- Order: Embedding → per layer [RMSNorm → fused QKV → RoPE →
  KV-cache update → GQA → Output projection → Residual → RMSNorm →
  MLP (swiglu + down) → Residual] → Final normalization → Logits →
  Argmax. **11 nodes per layer; `node_count_for(L) = 4 + 11·L`**
  (15 for L=1, 26 for L=2).
- Single-token decode only: `seq_len != 1` is rejected (Phase B
  embedding reads `tokens[0]`; `decode.metal` is single-token).
- Graph instantiation: FP16 activations throughout; fused QKV
  decodes Scheme C; output projection, MLP and logits decode
  Scheme A. Each node's `weight_format` is recorded and validated;
  the Scheme-B stack is ported as shaders + CPU refs but is not
  mixed into this graph (its kernels are float-activation; mixing
  would reinterpret FP16 tensors as float).
- Per-node descriptor + format validation, plus a queried-caps
  check: `query_capabilities()` reads the device's actual
  `vkGetPhysicalDeviceProperties` limits (workgroup counts /
  invocations / size, push-constant size, shared memory) when a
  device exists; on this machine caps are reported unknown with
  the BLOCKED reason, and the graph still validates against the
  documented defaults. No assumed NVIDIA features anywhere.
- `vk_dispatch graph [layers]` prints the node list with shader,
  format and workgroup counts and states explicitly that execution
  was not attempted. `vk_dispatch caps` prints queried limits or
  BLOCKED, exit 3.

## 4. Found defects and documented divergences

Metal defects found by reading the source (preserved, not repaired):

1. **RoPE K over-rotation** (`rope.metal` `rope_qk`): the K pair is
   rotated inside the per-Q-head loop, so each K pair is rotated
   `q_heads/kv_heads` times (3× at defaults). `decode.metal`
   rotates each K head once. Phase C ports rotate-once semantics
   (both `rope.comp` and the CPU ref) and documents the divergence.
   A dedicated test asserts K is rotated exactly once under GQA.
2. **Causal mask is a no-op** (`attention.metal` `causal_mask`):
   `if (pos > spec.offset) return;` precedes the only write, so it
   never masks. It is NOT ported as a kernel; causality is the
   softmax `len = offset+1` plus the zero tail.

Other divergences, recorded not silently unified:

- **Two RoPE layouts**: interleaved adjacent pairs (`rope.metal`)
  vs half-split pairs (d, d+head_dim/2) (`gqa_strided.metal`,
  `int4_transformer.metal`). Ported as separate shaders
  (`rope.comp`, `rope_split.comp`); non-interchangeable.
- **Two silu forms**: `mlp.metal` clamps the silu argument to
  [−80, 80]; `int4_transformer.metal` `swigluFFN` does not. Both
  ported (`swiglu.comp` clamped, `swiglu_act.comp` unclamped).
- **KV-cache dtype split**: `kv_cache.metal` caches are FP16;
  `int4_transformer.metal` `secureClearKV` operates on a **float**
  cache. Repo-level split, documented; the graph uses FP16 caches.
- **Fused QKV tile geometry**: the Metal kernel tiles 64×64
  simdgroup tiles; the GLSL port uses a shared X tile capped at
  K ≤ 3072 with different geometry and identical math (documented
  in the shader header).
- **Scheme C group rule**: C's scale-index math
  (`row*(K/64) + gidx`) is only well-defined when K % 64 == 0 —
  otherwise every row reads row 0's scale. The descriptor
  validation therefore refuses C when `in_dim % 64 != 0`. The
  graph test uses hidden = 64, the smallest admissible size.
- **`residual_and_norm`** computes the norm of x only, then adds
  the residual after scaling — as written, documented.
- `rmsnorm_ref` uses a double accumulator for the sum of squares;
  tolerances account for it. `attention.metal` uses
  `kv_head = head % kv_heads` (modulo) where `gqa_strided` uses
  division `h / (H/G)`; both ported as written.

## 5. Verification — actual results

```
ctest: 64 tests — 61 passed, 3 skipped (BLOCKED by design), 0 failed
  int4_oracle (Phase B): 31/31 PASS (incl. exhaustive roundtrip of
    all 65,536 half patterns; golden matvec within 5e-7)
  kernel_ref (Phase C): 83/83 PASS
  spirv-val: 27/27 shaders PASS
  reflection: 27/27 shaders PASS (bindings, push sizes, spec consts)
  golden reproducible: Phase B + Phase C goldens regenerate
    byte-identical
  dispatch: list/caps/embedding/matvec SKIP as BLOCKED
    (BLOCKED_NO_PHYSICAL_DEVICE, exit 3); graph/graph_2layers PASS
    (build + validate only, no execution)
```

CPU reference coverage in `test_kernel_ref.cpp` (83 checks):
dequant A/C exhaustive bitwise; `matvec_fmt` all 3 formats vs
double refs (incl. A in_dim=40 tail); descriptor validation
valid/invalid battery; format-compat refusal checks; byte
accounting (B: 128 packed + 32 scale for [4,64]; C: 256+16 for
[4,128]; A: 128+8); rmsnorm hand case + hidden=3072; layernorm
hand case; rope hand cases + K-once-under-GQA; kv roundtrip /
layer aliasing / clear; gqa strided hand case + online agreement
+ kv_len=1 identity; decomposed pipeline == fused; swiglu hand +
full; residual bitwise; lora hand cases + rank-65 refusal; argmax
ties; temperature softmax; decode_layer determinism (bitwise) +
cache checks; golden replay (rmsnorm + decode_layer, bitwise);
graph build / order / format / rejection tests (layers=0,
seq_len=2, non-multiple heads, odd head_dim).

Tolerances used: 1e-5 for norm/MLP/logits paths, 1e-6 for
attention/RoPE paths, exact bitwise where the spec is exact
(dequant, residual, argmax, golden replay, determinism). The
harness reports pass/fail at those tolerances; no observed-max
error is claimed beyond them.

Two test-construction defects were caught by the results and
fixed in-test (not in the code under test): the first decode_layer
golden used weights in [−0.5, 0.5), which quantize to all zeros
under Scheme A at scale 1.0 and made the composed output equal
the input — weights are now in [−4, 4); and the pos-1 cache
check originally reused the same x, for which the composed
output is genuinely identical (identical V rows), so it now uses
a different x at pos 1.

## 6. What is NOT claimed

- No GPU numerical equivalence: dispatch is BLOCKED on this
  machine (no ICD). The shaders are compiled, validated and
  reflection-checked; their math is verified against independent
  CPU references. Equivalence on hardware awaits a usable Vulkan
  device (Jetson).
- No performance claims of any kind.
- Phase B evidence is untouched: `int4_golden.txt` and all Phase B
  tests pass unchanged.

## 7. Files

- `Vulkan/include/tensor_abi.h`, `Vulkan/src/tensor_abi.cpp` (new)
- `Vulkan/include/kernel_ref.h`, `Vulkan/src/kernel_ref.cpp` (new)
- `Vulkan/include/command_graph.h`, `Vulkan/src/command_graph.cpp` (new)
- `Vulkan/include/int4_abi.h`, `Vulkan/src/int4_oracle.cpp` (extended:
  A/C decode, quantize, `matvec_fmt`)
- `Vulkan/include/vulkan_runtime.h`, `Vulkan/src/vulkan_runtime.cpp`
  (extended: `DeviceCaps`, `query_capabilities()`)
- `Vulkan/shaders/`: 25 new `.comp` + checked-in `.spv` (27 total)
- `Vulkan/tests/test_kernel_ref.cpp`, `gen_kernel_golden.cpp`,
  `golden/kernel_golden.txt` (new); `check_reflection.cpp`
  (rewritten table-driven, 27 shaders); `test_int4_oracle.cpp`
  (unchanged); `vk_dispatch.cpp` (new `caps` + `graph` modes)
- `Vulkan/CMakeLists.txt` (new libraries, 27-shader loop, per-shader
  spirv-val + reflection tests, kernel golden + dispatch tests)
