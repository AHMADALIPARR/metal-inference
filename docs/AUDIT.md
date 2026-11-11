# Phase A — Source Audit (measured 2026-11-11)

Scope: complete tree of SNAPKITTYWEST/metal-inference @ a703d72, plus
mamba-metal @ d48a698 as structural reference. Method: direct source
inspection and execution-environment probes. README claims were not
treated as evidence.

## Inventory

- Sources/Kernels: 17 .metal files, 10,240 lines total tree 12,065.
  - Transformer.metal (2,331 lines, 60+ kernel entry points) — the
    integrated kernel set; the Swift host loads this file as its
    library resource.
  - apple_metal.metal (7,200 lines, 561 kernel entry points) — real
    kernel bodies (spot-checked rmsnorm_kernel: full reduction loop),
    largely naive per-thread formulations, heavily redundant with
    the per-op files. Not stubs, but not the shipping path either:
    nothing references it from the host.
  - 15 per-op files (29–271 lines): embedding, normalization,
    fused_qkv_tiled (fused_qkv_int4), rope, gqa_strided, kv_cache,
    attention, matmul, linear, mlp, residual, decode (decode_step),
    output, lora_delta, int4_transformer (9 kernels: qkvProjection
    et al.). Bodies are real implementations.
- Sources/Host: MetalTransformer.swift (40 lines), int4_host_runtime.mm
  (327 lines — entropy routing + AES-GCM + audit tests:
  testKVCachesCleared, testDirectiveInjectionBlocked,
  testBoundaryChecks, runAuditSuite), block_layout.h, adapter.h,
  adapter_slot.h, kv_test.h, safety.h.
- Sources/Gate: entropy routing in Python (6 files, 143 lines total);
  the same routing logic also exists in C++ inside
  int4_host_runtime.mm (routeDecision, logitEntropy).
- Sources/M5Gateway: M5OSApp.swift (698 lines) + m5_gateway.wat
  (93 lines).
- Sources/Common/types.h: ModelConfig + index helpers (kv_offset,
  qkv_index, attention_scale = 1/sqrt(head_dim)).
- Tests: one XCTest (11 lines): device presence + pipeline load +
  headDim == 128. No numeric/golden tests exist anywhere.
- Model defaults (MetalTransformer.swift TransformerDimensions,
  matches the directive): hidden 3072, intermediate 8192, heads 24,
  kv_heads 8, head_dim 128, vocab 32000, context 4096.

## Existing build status: BROKEN in three independent places

1. Package.swift declares target path "Sources/MetalTransformer",
   which does not exist in the tree. The package cannot build as
   committed, on any platform.
2. MetalTransformer.swift requests pipeline functions by name —
   qmatvec_i4, qkv_projection_i4, causal_softmax, attention_times_v,
   output_projection_i4, copy_kv, final_normalization, rope — that
   do not exist under those names in Transformer.metal (which
   defines quantized_matvec, qkv_projection, softmax_stable,
   attention_value, logits, kv_store, final_norm, rope_qk). Init
   would throw on a Mac.
3. The forward pass is not wired: the Swift host exposes only
   encodeToken (embedding) and greedyToken (CPU argmax over FP16
   logits readback). No encode/decode pipeline exists host-side.

## INT4 ABI — measured from int4_transformer.metal (the only definition)

The directive assumed block_layout.h defines the quantization
layout. It does not: block_layout.h is a KV-cache block allocator
(superblock 8; 5 dedicated + 3 shared blocks). Separately flagged:
its shared_kv_bytes() returns one block's bytes while
dedicated_kv_bytes() returns five, and total_kv_bytes() multiplies
(dedicated + shared) by superblock count — SHARED_BLOCKS=3 is never
used. Intent unverified; formula suspect.

Actual quantization contract (dequantNibble, int4_transformer.metal):
- Group size 32; packing 2 weights/byte; exactly 16 packed bytes
  per group.
- Nibble order: even index -> LOW nibble, odd index -> HIGH nibble.
- Dequantization: w = (nib - 7.5) * scale. Unsigned nibble 0..15,
  half-integer offset 7.5, NO zero-point field, NO sign extension.
- Scales: device const half2* indexed [group].x — one FP16 scale
  per group carried in the .x lane; .y is not read by this
  function. Host-side size check in int4_host_runtime.mm
  (scaleBytes >= outDim * (inDim/32) * sizeof(uint16_t)) is
  consistent with one FP16 per group.
- qkvProjection asserts hidden even and hidden % 32 == 0 via a
  fault-counter bassert (atomics present).

## Dependency graph (forward pass, per-op files)

embedding -> rmsnorm -> fused_qkv_int4 -> rope -> kv_cache store ->
gqa_strided (reads kv_cache) -> attention -> linear (output proj)
-> residual -> rmsnorm -> mlp (swiglu + down) -> residual -> (x N
layers) -> final_norm (normalization) -> output (logits) ->
token_select; lora_delta adds into projections; decode_step is the
single-token integration entry. Gate routing consumes final logits
(entropy) host-side; M5 gateway governs buffer admission.

## Environment — measured on the audit machine (Muse VM)

- Present: cmake, clang++, g++, OCaml (micromamba env).
- Absent: swift, glslangValidator, glslc, spirv-val, vulkaninfo;
  NO Vulkan ICD installed — Vulkan execution is impossible here.
- Jetson Orin NX (snapkitty-jetson): SSH connection reset — the
  host is down (unchanged since 2026-11-09; needs a physical
  power-cycle). RTX 3080 rig: in storage.
- Consequence per the directive's own rules: shader compilation
  and SPIR-V validation become possible after installing
  glslang/spirv-tools (conda-forge); GPU execution, cross-backend
  comparison, and all hardware benchmarks (Phases B-verify, D, F)
  are BLOCKED until the Jetson is powered on. No CPU result will
  be labeled a GPU result.

## Push destination

SNAPKITTYWEST is an org the owner's AHMADALIPARR key cannot write
(403, verified previously). All work proceeds in this local clone;
the landing repository is the owner's decision.

## Addendum (2026-11-11, Phase B preparation) — correction to the INT4 section

The audit above described int4_transformer.metal's scheme as the
repo's INT4 definition. That was incomplete. The repo contains
THREE mutually incompatible INT4 dequantization schemes:

- Scheme A (signed): Transformer.metal dequant_i4 (line 69) and
  apple_metal.metal dequant_i4_signed — nibble read as two's
  complement (nib >= 8 -> nib - 16, range -8..7). Transformer.metal
  applies a PER-ROW scale (scale[row], flat half*).
- Scheme A' (signed, per-group): fused_qkv_tiled.metal
  dequant_int4 — same signed nibble, but scales are per-group,
  indexed S[row * (K/32) + group] in a flat half array.
- Scheme B (offset): int4_transformer.metal dequantNibble —
  unsigned nibble, (nib - 7.5) * scale, per-group scale stored as
  half2 and read at [group].x.

Nibble order agrees everywhere (even index = low nibble). Value
mapping and scale granularity do not. A weight blob packed for one
scheme decodes incorrectly under the others. Phase B implements
Scheme B exactly, per the owner's directive; the Vulkan ABI
document names the scheme it implements and the two it does not.
