# Phase B — Report (executed 2026-11-11)

Scope as approved: functional vertical slice in the local clone on
branch feat/vulkan-int4-phase-b. The Apple/Swift side is untouched;
the Package.swift path mismatch and host kernel-symbol mismatch
remain documented in docs/AUDIT.md, deliberately unrepaired.

## What was built

- Vulkan/include/int4_abi.h — the Scheme-B INT4 ABI as a written
  contract: group 32, 2 weights/byte (16 bytes/group), low nibble
  first, w = (nib - 7.5) * scale, one FP16 scale per group stored
  in the .x lane of a half2 slot, no zero point, no sign extension.
  Names the two repo schemes it is NOT (docs/AUDIT.md addendum).
- Vulkan/src/int4_oracle.cpp — CPU oracle: FP16<->FP32 with
  round-to-nearest-even, quantize (oracle-defined RNE rule; the
  repo defines no quantizer), exact dequantize, serial-order
  matvec (the Metal kernels' accumulation order).
- Vulkan/shaders/embedding.comp — manual GLSL 450 port of
  embedding.metal's `embedding`. FP16 values move as bit patterns
  in uint buffers; the copy is bitwise.
- Vulkan/shaders/int4_matvec.comp — manual GLSL 450 port of the
  Scheme-B matvec (dequantNibble math in quantized_matvec's shape):
  one workgroup per row, shared-memory reduction. Reduction order
  differs from the serial Metal loop; the dispatch comparison uses
  a stated tolerance for exactly that reason.
- Vulkan/src/vulkan_runtime.cpp + tools/vk_dispatch.cpp — C++17
  runtime: dlopen'd loader, instance/device creation, compute
  queue discovery, host-visible buffers, descriptor sets, compute
  pipelines with specialization + push constants, command buffer
  dispatch, fence synchronization, readback, full cleanup. Every
  failure is a typed Status (LoaderMissing, NoPhysicalDevice, ...).
  No CPU fallback exists in this code path.
- Vulkan/tests/ — test_int4_oracle.cpp (31 checks), gen_golden.cpp
  + checked-in tests/golden/int4_golden.txt, check_reflection.cpp
  (SPIRV-Reflect), case_data.h (one deterministic case shared by
  generator, oracle test, and dispatch CLI).

These two shaders are manually ported reference implementations.
No Metal-to-Vulkan source translation (compiler) is claimed.

## ABI mapping (Metal -> Vulkan)

embedding: buffer(0) table -> binding 0 (uint, FP16 bits);
buffer(1) token_ids -> binding 1; buffer(2) output -> binding 2;
ModelConfig -> push constants { hidden : u32 }.
int4_matvec: packed weights -> binding 0 (uint array; byte k of a
row = lanes 2k low / 2k+1 high, identical to the Metal uchar
layout); half2 scale slots -> binding 1 (uint per slot, .x in the
low 16 bits); float x -> binding 2; float y -> binding 3; push
constants { in_dim, out_dim }; spec constants 0 = workgroup size,
1 = group size (32).

## Results (actual output)

CPU oracle — 31 passed, 0 failed:
- FP16 constants incl. RNE boundaries (65520 -> inf, 2^-25 tie ->
  0, 3*2^-25 tie -> 2^-23), exhaustive roundtrip over all 65,536
  half patterns.
- Dequantization exhaustive: 16 nibbles x 7 scales, bitwise equal
  to the double-precision-derived value; boundary values -7.5s,
  +7.5s, -0.5s, +0.5s exact.
- Packing: geometry (32->16 B, 33->17 B tail), nibble order,
  quantize/dequant roundtrip exact, saturation, tail-group matvec
  vs double reference.
- Golden vectors: dequant_row 64 lanes bitwise; matvec within
  5e-7 (summation-order rounding only).

Shaders — glslangValidator (Vulkan 1.2 target) compiles both;
spirv-val passes both. Reflection (SPIRV-Reflect): entry point,
descriptor layout (set 0, sequential, storage buffers), push
block sizes (4 B / 8 B), spec constants — all assertions PASS
(6/6 embedding, 7/7 matvec).

CTest — 9 tests: 7 passed, 2 skipped, 0 failed.
- Passed: int4_oracle, golden_reproducible (regeneration is
  byte-identical), spirv_val x2, reflect x2, dispatch_list.
- Skipped (by design): dispatch_embedding, dispatch_matvec.

Vulkan dispatch — BLOCKED, not passed:
  vk_dispatch list -> "BLOCKED: BLOCKED_NO_PHYSICAL_DEVICE —
  vkCreateInstance failed: the Vulkan loader found no usable
  ICD/driver on this system" (exit 3).
The Vulkan loader (libvulkan 1.3.275) is installed on this VM;
no ICD is. Two attempts to install Mesa's Lavapipe software ICD
(apt, then direct .deb from the Ubuntu pool) failed on mirror
reachability. No CPU result is presented as a Vulkan result.
GPU execution remains unverified until a device with a Vulkan
driver is available to this code (Jetson Orin NX or RTX 3080).

## Reproduce

  export VULKAN_ENV=$HOME/workspace/.tools/vulkan-env  # glslang, spirv-tools, headers, SPIRV-Reflect
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --parallel
  ctest --test-dir build --output-on-failure
  build/Vulkan/vk_dispatch list        # device census or BLOCKED

Checked-in SPIR-V binaries in Vulkan/shaders/ were produced by
the CMake rules above (glslangValidator -V --target-env vulkan1.2).

## Acceptance

CPU oracle passes its numeric tests: YES.
Both shader modules compile, validate, pass reflection: YES.
GPU inference execution: UNVERIFIED (hardware/ICD unavailable).
