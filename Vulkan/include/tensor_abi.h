// tensor_abi.h — canonical tensor descriptor and quantization-format
// identity for the Vulkan backend (Phase C).
//
// The repo's three INT4 schemes are NOT interchangeable. Every weight
// tensor carries its format explicitly, and no kernel launches until
// the format check for that kernel passes. There is no implicit
// default format anywhere in this subsystem.
//
//   Format A (Scheme A, Transformer.metal dequant_i4,
//             Sources/Common/types.h dequant_i4):
//     signed nibble, value -8..7 (nib >= 8 -> nib - 16)
//     scale: one FP16 PER ROW, flat half* scale[row]
//     group: none (group_size = 0)
//     packing: row_bytes = ceil(cols/2), even lane -> low nibble
//
//   Format B (Scheme B, int4_transformer.metal dequantNibble):
//     unsigned nibble, w = (nib - 7.5) * scale
//     scale: one FP16 PER GROUP of 32, stored as .x of a half2 slot
//            (.y carried, never read)
//     group_size = 32
//     packing: 2 weights/byte, 16 packed bytes/group, low nibble first
//
//   Format C (Scheme C, fused_qkv_tiled.metal dequant_int4):
//     signed nibble, value -8..7 (same signed mapping as A)
//     scale: one FP16 PER GROUP of 64, flat half* layout,
//            index = row * (K / 64) + group
//     group_size = 64  (GROUP_SIZE in that file; NOT 32)
//     packing: byte k of a row holds lanes 2k (low) and 2k+1 (high)
//
// A blob packed under one format decodes to different weights under
// another. The descriptor is what stops that from happening silently.
#pragma once
#include <cstdint>
#include <cstddef>
#include <string>

namespace tensorabi {

enum class DType : uint32_t { F16 = 0, F32 = 1, U8 = 2, U32 = 3 };

inline constexpr size_t dtype_bytes(DType d) {
  switch (d) {
    case DType::F16: return 2;
    case DType::F32: return 4;
    case DType::U8: return 1;
    case DType::U32: return 4;
  }
  return 0;
}

inline const char* dtype_name(DType d) {
  switch (d) {
    case DType::F16: return "f16";
    case DType::F32: return "f32";
    case DType::U8: return "u8";
    case DType::U32: return "u32";
  }
  return "?";
}

enum class QuantFormat : uint32_t {
  None = 0,                 // dense tensor, no quantization
  A_SignedPerRow = 1,       // Scheme A
  B_Offset75PerGroup32 = 2, // Scheme B
  C_SignedPerGroup64 = 3,   // Scheme C
};

inline const char* format_name(QuantFormat f) {
  switch (f) {
    case QuantFormat::None: return "none";
    case QuantFormat::A_SignedPerRow: return "A(signed,per-row)";
    case QuantFormat::B_Offset75PerGroup32: return "B((nib-7.5),group32,half2)";
    case QuantFormat::C_SignedPerGroup64: return "C(signed,group64,flat)";
  }
  return "?";
}

// Group size the format demands (0 = no grouping, per-row scale).
inline constexpr uint32_t format_group_size(QuantFormat f) {
  switch (f) {
    case QuantFormat::None: return 0;
    case QuantFormat::A_SignedPerRow: return 0;
    case QuantFormat::B_Offset75PerGroup32: return 32;
    case QuantFormat::C_SignedPerGroup64: return 64;
  }
  return 0;
}

// Scale storage layout tag. 0 = flat FP16, one per row (A).
// 1 = flat FP16, one per group (C). 2 = half2 slots, .x = scale (B).
inline constexpr uint32_t format_scale_layout(QuantFormat f) {
  switch (f) {
    case QuantFormat::None: return 0;
    case QuantFormat::A_SignedPerRow: return 0;
    case QuantFormat::B_Offset75PerGroup32: return 2;
    case QuantFormat::C_SignedPerGroup64: return 1;
  }
  return 0;
}

// Scale slots (count, not bytes) a weight row of in_dim lanes needs.
inline constexpr uint64_t scale_slots_per_row(QuantFormat f, uint32_t in_dim) {
  switch (f) {
    case QuantFormat::None: return 0;
    case QuantFormat::A_SignedPerRow: return 1;
    case QuantFormat::B_Offset75PerGroup32: return (in_dim + 31u) / 32u;
    case QuantFormat::C_SignedPerGroup64: return (in_dim + 63u) / 64u;
  }
  return 0;
}

// Packed data bytes a weight row of in_dim lanes needs (all three
// formats pack 2 nibbles per byte).
inline constexpr uint64_t packed_row_bytes(uint32_t in_dim) {
  return (in_dim + 1u) / 2u;
}

struct TensorDescriptor {
  DType dtype = DType::F32;
  uint32_t rank = 0;
  uint32_t shape[4] = {0, 0, 0, 0};   // elements per axis, row-major
  uint32_t strides[4] = {0, 0, 0, 0}; // elements between axis entries
  QuantFormat format = QuantFormat::None;
  uint32_t group_size = 0;      // must equal format_group_size(format)
  DType scale_dtype = DType::F16;
  uint32_t scale_layout = 0;    // must equal format_scale_layout(format)
  uint64_t buffer_offset = 0;   // bytes into the owning buffer
  uint32_t alignment = 16;      // required byte alignment of buffer_offset

  uint64_t num_elements() const {
    if (rank == 0) return 0;
    uint64_t n = 1;
    for (uint32_t i = 0; i < rank; ++i) n *= shape[i];
    return n;
  }
  // Bytes of the (packed) data region, excluding scales. For a
  // quantized weight, shape is in logical lanes [rows, in_dim] but
  // storage is packed 2 nibbles/byte: rows * ceil(in_dim/2).
  uint64_t data_bytes() const {
    if (format != QuantFormat::None && rank == 2)
      return (uint64_t)shape[0] * packed_row_bytes(shape[1]);
    return num_elements() * dtype_bytes(dtype);
  }
  // Bytes of the scale region for a 2-D weight [rows, in_dim].
  // rows = shape[0], in_dim = shape[1]. Non-weight tensors: 0.
  uint64_t scale_bytes() const {
    if (format == QuantFormat::None || rank < 2) return 0;
    uint64_t slots = (uint64_t)shape[0] * scale_slots_per_row(format, shape[1]);
    // Format B stores half2 slots (4 bytes each); A and C store FP16 (2).
    return slots * (format == QuantFormat::B_Offset75PerGroup32 ? 4u : 2u);
  }
};

// Validate a descriptor against the buffer that will hold it.
// buffer_bytes is the whole owning buffer's size. On failure returns
// false and a human-readable reason (never throws). Checks:
//   rank 1..4, every shape dim > 0, strides describe a row-major
//   contiguous layout (strides[i] == product of shape[i+1..]),
//   buffer_offset aligned, offset + data_bytes (+ scale_bytes for
//   weights) inside the buffer, format/group/layout/scale-dtype
//   consistency, and the format's divisibility rules for a 2-D
//   weight (in_dim even for all formats; B needs in_dim % 32 == 0,
//   C needs in_dim % 64 == 0 — the Metal kernels assert the same).
bool validate_descriptor(const TensorDescriptor& d, uint64_t buffer_bytes,
                         std::string& why);

// A kernel declares the weight format it decodes. A launch is legal
// only when the tensor's format is exactly that format. Returns false
// with a reason on mismatch; there is no fallback and no silent
// reinterpretation.
bool check_format_compatible(const TensorDescriptor& weight,
                             QuantFormat expected, std::string& why);

// Row-major contiguous strides for a shape (helper for builders).
inline void contiguous_strides(uint32_t rank, const uint32_t* shape,
                               uint32_t* strides_out) {
  uint32_t s = 1;
  for (int i = (int)rank - 1; i >= 0; --i) {
    strides_out[i] = s;
    s *= shape[i];
  }
}

} // namespace tensorabi
