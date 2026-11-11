// int4_abi.h — the INT4 tensor ABI implemented by this subsystem.
//
// SCHEME IDENTITY. This repository contains three incompatible INT4
// dequantization schemes (see docs/AUDIT.md addendum). This ABI is
// Scheme B, measured from Sources/Kernels/int4_transformer.metal
// (dequantNibble, kQuantGroup, kInt4Offset):
//
//   group size        32 weights
//   packing           2 weights per byte, 16 packed bytes per group
//   nibble order      even index -> LOW nibble, odd index -> HIGH
//   value mapping     w = (nib - 7.5) * scale, nib unsigned 0..15
//   scale             one FP16 per group, stored as the .x lane of a
//                     half2 slot (the .y lane is carried, not read)
//   zero point        none (the 7.5 offset replaces it)
//   sign extension    none (nibbles are unsigned)
//
// It is NOT Transformer.metal's scheme (signed nibble, per-row
// scale) and NOT fused_qkv_tiled.metal's scheme (signed nibble,
// per-group flat scale). Blobs are not interchangeable between
// schemes.
//
// The repository defines no quantizer (float -> nibble). The
// oracle's quantizer is defined here: nib = clamp(RNE(w/scale +
// 7.5), 0, 15), RNE = round-to-nearest-even. Dequantization — the
// direction the kernels define — is exact and is what the golden
// vectors pin down.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace int4abi {

inline constexpr uint32_t kGroupSize = 32;
inline constexpr float kOffset = 7.5f;
inline constexpr uint32_t kPackedBytesPerGroup = kGroupSize / 2; // 16

// ---- FP16 <-> FP32, round-to-nearest-even on narrowing ----
float half_to_float(uint16_t h);
uint16_t float_to_half(float f);

// ---- packing geometry ----
// Bytes occupied by one packed row of in_dim weights.
inline constexpr size_t packed_row_bytes(uint32_t in_dim) {
  return (in_dim + 1) / 2;
}
// Scale slots (half2) occupied by one row.
inline constexpr size_t scale_slots_per_row(uint32_t in_dim) {
  return (in_dim + kGroupSize - 1) / kGroupSize;
}

// Extract lane's nibble from a packed row (low-nibble-first).
inline uint32_t nibble_at(const uint8_t* packed_row, uint32_t lane) {
  uint8_t b = packed_row[lane >> 1];
  return (lane & 1u) ? (b >> 4) : (b & 0x0Fu);
}

// The ABI's dequantization, exactly as the Metal source computes it:
// (float(nib) - 7.5f) * float(scale_half). half->float is exact, the
// subtraction is exact (results are k/2 for integer k in [-15, 15]),
// the product is one FP32 rounding.
float dequant_nibble(uint32_t nib, uint16_t scale_half);

// Dequantize one element of a packed row by lane index.
float dequant_at(const uint8_t* packed_row, const uint32_t* scale_slots,
                 uint32_t lane);

// Quantize one row of floats into packed nibbles + half2 scale
// slots (.x = scale, .y = 0). Scales are supplied by the caller per
// group (the repo fixes no scale-selection rule). Lanes past in_dim
// in a tail group are never written by matvec readers; the packer
// zero-fills pad nibbles.
void quantize_row(const float* w, uint32_t in_dim, const uint16_t* group_scales,
                  uint8_t* packed_out, uint32_t* scale_slots_out);

// Reference matvec, serial accumulation order (the Metal kernels'
// order): y[r] = sum_c dequant(w[r,c]) * x[c]. x and y are FP32.
void matvec(const uint8_t* packed, const uint32_t* scale_slots,
            const float* x, float* y, uint32_t in_dim, uint32_t out_dim);

} // namespace int4abi
