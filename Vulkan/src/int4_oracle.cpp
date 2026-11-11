// int4_oracle.cpp — CPU oracle for the Scheme-B INT4 ABI.
// No Python, no external dependencies. This is the reference the
// SPIR-V kernels are checked against; it is not a GPU result.
#include "int4_abi.h"
#include <cstring>
#include <cmath>

namespace int4abi {

float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h >> 15) << 31;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t frac = h & 0x3FFu;
  uint32_t out;
  if (exp == 0) {
    if (frac == 0) {
      out = sign; // +/- zero
    } else {
      // subnormal half -> normalized float
      int e = -14;
      uint32_t f = frac;
      while ((f & 0x400u) == 0) { f <<= 1; --e; }
      f &= 0x3FFu;
      out = sign | ((uint32_t)(e + 127) << 23) | (f << 13);
    }
  } else if (exp == 0x1Fu) {
    out = sign | 0x7F800000u | (frac << 13); // inf / nan
  } else {
    out = sign | ((exp - 15 + 127) << 23) | (frac << 13);
  }
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

uint16_t float_to_half(float fv) {
  uint32_t x;
  std::memcpy(&x, &fv, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int exp = (int)((x >> 23) & 0xFFu) - 112; // rebased for half
  uint32_t frac = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFFu) == 0xFFu) {
    // inf / nan
    return (uint16_t)(sign | 0x7C00u | (frac ? 0x200u : 0u));
  }
  if (exp >= 31) return (uint16_t)(sign | 0x7C00u); // overflow -> inf (RNE boundary handled below via mantissa path when exp==30.x)
  if (exp <= 0) {
    // subnormal half or zero
    if (exp < -10) return (uint16_t)sign; // underflow to zero
    frac |= 0x800000u; // implicit leading 1
    int shift = 14 - exp; // bits to drop: 13 + (1 - exp)
    uint32_t half_frac = frac >> shift;
    uint32_t rem = frac & ((shift == 32) ? 0xFFFFFFFFu : ((1u << shift) - 1u));
    uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half_frac & 1u))) ++half_frac;
    return (uint16_t)(sign | half_frac);
  }
  // normal half: drop 13 mantissa bits with round-to-nearest-even
  uint32_t out = sign | ((uint32_t)exp << 10) | (frac >> 13);
  uint32_t rem = frac & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (out & 1u))) ++out;
  return (uint16_t)out;
}

float dequant_nibble(uint32_t nib, uint16_t scale_half) {
  return ((float)nib - kOffset) * half_to_float(scale_half);
}

float dequant_at(const uint8_t* packed_row, const uint32_t* scale_slots,
                 uint32_t lane) {
  uint32_t group = lane / kGroupSize;
  uint16_t scale = (uint16_t)(scale_slots[group] & 0xFFFFu); // half2 .x lane
  return dequant_nibble(nibble_at(packed_row, lane), scale);
}

void quantize_row(const float* w, uint32_t in_dim, const uint16_t* group_scales,
                  uint8_t* packed_out, uint32_t* scale_slots_out) {
  size_t bytes = packed_row_bytes(in_dim);
  for (size_t i = 0; i < bytes; ++i) packed_out[i] = 0;
  uint32_t groups = (uint32_t)scale_slots_per_row(in_dim);
  for (uint32_t g = 0; g < groups; ++g)
    scale_slots_out[g] = (uint32_t)group_scales[g]; // .x = scale, .y = 0
  for (uint32_t lane = 0; lane < in_dim; ++lane) {
    float scale = half_to_float(group_scales[lane / kGroupSize]);
    float t = (scale == 0.0f) ? kOffset : (w[lane] / scale + kOffset);
    long n = (long)std::nearbyint(t); // nearbyint rounds half-to-even (default FP mode)
    if (n < 0) n = 0;
    if (n > 15) n = 15;
    uint8_t& b = packed_out[lane >> 1];
    if (lane & 1u) b |= (uint8_t)(n << 4);
    else b |= (uint8_t)(n & 0x0Fu);
  }
}

void matvec(const uint8_t* packed, const uint32_t* scale_slots,
            const float* x, float* y, uint32_t in_dim, uint32_t out_dim) {
  size_t row_bytes = packed_row_bytes(in_dim);
  size_t row_slots = scale_slots_per_row(in_dim);
  for (uint32_t r = 0; r < out_dim; ++r) {
    const uint8_t* prow = packed + r * row_bytes;
    const uint32_t* srow = scale_slots + r * row_slots;
    float acc = 0.0f;
    for (uint32_t c = 0; c < in_dim; ++c)
      acc += dequant_at(prow, srow, c) * x[c];
    y[r] = acc;
  }
}

// ---- Phase C: Schemes A and C (separate decode paths) ----

float dequant_a_at(const uint8_t* packed, const uint16_t* row_scales,
                   uint32_t row, uint32_t lane, uint32_t in_dim) {
  size_t row_bytes = packed_row_bytes(in_dim);
  const uint8_t* prow = packed + (size_t)row * row_bytes;
  uint32_t nib = nibble_at(prow, lane);
  return (float)signed_nibble_value(nib) * half_to_float(row_scales[row]);
}

float dequant_c_at(const uint8_t* packed, const uint16_t* group_scales,
                   uint32_t row, uint32_t lane, uint32_t in_dim) {
  size_t row_bytes = packed_row_bytes(in_dim);
  uint32_t groups = (in_dim + 63u) / 64u;
  const uint8_t* prow = packed + (size_t)row * row_bytes;
  uint32_t nib = nibble_at(prow, lane);
  return (float)signed_nibble_value(nib) *
         half_to_float(group_scales[(size_t)row * groups + lane / 64u]);
}

void quantize_row_a(const float* w, uint32_t in_dim, uint16_t row_scale,
                    uint8_t* packed_out) {
  size_t bytes = packed_row_bytes(in_dim);
  for (size_t i = 0; i < bytes; ++i) packed_out[i] = 0;
  float scale = half_to_float(row_scale);
  for (uint32_t lane = 0; lane < in_dim; ++lane) {
    long v = (scale == 0.0f) ? 0 : (long)std::nearbyint((double)w[lane] / scale);
    if (v < -8) v = -8;
    if (v > 7) v = 7;
    uint8_t nib = (uint8_t)(v & 0x0F);
    if (lane & 1u) packed_out[lane >> 1] |= (uint8_t)(nib << 4);
    else packed_out[lane >> 1] |= nib;
  }
}

void quantize_row_c(const float* w, uint32_t in_dim, const uint16_t* group_scales,
                    uint8_t* packed_out) {
  size_t bytes = packed_row_bytes(in_dim);
  for (size_t i = 0; i < bytes; ++i) packed_out[i] = 0;
  for (uint32_t lane = 0; lane < in_dim; ++lane) {
    float scale = half_to_float(group_scales[lane / 64u]);
    long v = (scale == 0.0f) ? 0 : (long)std::nearbyint((double)w[lane] / scale);
    if (v < -8) v = -8;
    if (v > 7) v = 7;
    uint8_t nib = (uint8_t)(v & 0x0F);
    if (lane & 1u) packed_out[lane >> 1] |= (uint8_t)(nib << 4);
    else packed_out[lane >> 1] |= nib;
  }
}

void matvec_fmt(tensorabi::QuantFormat format, const uint8_t* packed,
                const uint16_t* scales, const float* x, float* y,
                uint32_t in_dim, uint32_t out_dim) {
  size_t row_bytes = packed_row_bytes(in_dim);
  uint32_t groups_b = (in_dim + 31u) / 32u;
  for (uint32_t r = 0; r < out_dim; ++r) {
    const uint8_t* prow = packed + (size_t)r * row_bytes;
    float acc = 0.0f;
    for (uint32_t c = 0; c < in_dim; ++c) {
      float w = 0.0f;
      switch (format) {
        case tensorabi::QuantFormat::A_SignedPerRow:
          w = dequant_a_at(packed, scales, r, c, in_dim);
          break;
        case tensorabi::QuantFormat::B_Offset75PerGroup32: {
          // B scales arrive as half2 slots viewed as uint16 pairs.
          const uint32_t* slots = reinterpret_cast<const uint32_t*>(scales);
          w = dequant_at(prow, slots + (size_t)r * groups_b, c);
          break;
        }
        case tensorabi::QuantFormat::C_SignedPerGroup64:
          w = dequant_c_at(packed, scales, r, c, in_dim);
          break;
        default:
          w = 0.0f; // QuantFormat::None has no decode path; never called.
          break;
      }
      acc += w * x[c];
    }
    y[r] = acc;
  }
}

} // namespace int4abi
