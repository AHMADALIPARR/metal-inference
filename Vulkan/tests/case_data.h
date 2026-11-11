// case_data.h — one deterministic matvec case shared by the golden
// generator, the oracle test, and the Vulkan dispatch CLI, so all
// three verify against byte-identical inputs.
#pragma once
#include "int4_abi.h"
#include <cstdint>
#include <vector>
#include <cmath>

namespace casedata {

struct Lcg {
  uint32_t s = 0x12345678u;
  uint32_t next() { s = s * 1664525u + 1013904223u; return s; }
  float next_sym() { // [-1, 1)
    return (float)((next() >> 8) & 0xFFFFFF) / 8388608.0f - 1.0f;
  }
};

struct MatvecCase {
  uint32_t in_dim = 64, out_dim = 4;
  std::vector<uint8_t> packed;
  std::vector<uint32_t> scales; // half2 slots
  std::vector<float> x;
  std::vector<double> expected_y; // double-precision reference
};

inline MatvecCase build_matvec_case() {
  MatvecCase c;
  Lcg rng;
  uint32_t groups = (c.in_dim + 31) / 32;
  std::vector<uint16_t> gscales(groups);
  const uint16_t scale_choices[2] = {0x3C00, 0x3400}; // 1.0, 0.25
  c.packed.resize((size_t)c.out_dim * int4abi::packed_row_bytes(c.in_dim));
  c.scales.resize((size_t)c.out_dim * groups);
  c.x.resize(c.in_dim);
  for (auto& v : c.x) v = rng.next_sym();
  std::vector<float> w(c.in_dim);
  for (uint32_t r = 0; r < c.out_dim; ++r) {
    for (uint32_t g = 0; g < groups; ++g) gscales[g] = scale_choices[(r + g) & 1];
    for (auto& v : w) v = rng.next_sym();
    int4abi::quantize_row(w.data(), c.in_dim, gscales.data(),
                          c.packed.data() + (size_t)r * int4abi::packed_row_bytes(c.in_dim),
                          c.scales.data() + (size_t)r * groups);
  }
  // double-precision reference from the packed representation
  c.expected_y.assign(c.out_dim, 0.0);
  for (uint32_t r = 0; r < c.out_dim; ++r) {
    const uint8_t* prow = c.packed.data() + (size_t)r * int4abi::packed_row_bytes(c.in_dim);
    const uint32_t* srow = c.scales.data() + (size_t)r * groups;
    double acc = 0.0;
    for (uint32_t col = 0; col < c.in_dim; ++col) {
      double dq = (double)int4abi::dequant_at(prow, srow, col);
      acc += dq * (double)c.x[col];
    }
    c.expected_y[r] = acc;
  }
  return c;
}

} // namespace casedata
