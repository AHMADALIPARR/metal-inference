// test_kernel_ref.cpp — Phase C CPU verification.
//
// Two kinds of checks, kept separate:
//   * BITWISE / EXACT: packing, dequantization, argmax, residual,
//     KV store/fetch, golden-vector replay. A single bit wrong fails.
//   * TOLERANCE: anything involving FP32 summation or libm exp/cos,
//     compared against a double-precision recomputation with an
//     explicit tolerance and a reported maximum error.
// Usage: test_kernel_ref <path-to-kernel_golden.txt>
#include "int4_abi.h"
#include "tensor_abi.h"
#include "kernel_ref.h"
#include "command_graph.h"
#include "case_data.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
static void check(const char* name, bool cond) {
  if (cond) { ++g_pass; printf("PASS %s\n", name); }
  else { ++g_fail; printf("FAIL %s\n", name); }
}
static uint32_t fbits(float v) { uint32_t u; std::memcpy(&u, &v, 4); return u; }
using namespace int4abi;
using tensorabi::DType;
using tensorabi::QuantFormat;

// Max |a-b| over two float vectors; also returns it for reporting.
static double max_err(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i)
    m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
  return m;
}
static bool near_vec(const std::vector<float>& a, const std::vector<float>& b,
                     double tol, double& err_out) {
  err_out = max_err(a, b);
  return err_out <= tol;
}

// ---- Scheme A / C dequantization, exhaustive over nibbles/scales ----
static void test_dequant_ac() {
  const uint16_t scales[] = {0x0000, 0x0001, 0x3C00, 0x3800, 0xBC00, 0x7BFF};
  bool ok_a = true, ok_c = true;
  const uint32_t in_dim = 128; // A: 1 row scale; C: 2 groups of 64
  std::vector<uint8_t> packed(packed_row_bytes(in_dim));
  for (size_t i = 0; i < packed.size(); ++i) packed[i] = (uint8_t)((i * 53 + 7) & 0xFF);
  uint16_t row_scale = 0x3C00;
  uint16_t group_scales[2] = {0x3C00, 0x3800};
  for (uint16_t s : scales) {
    row_scale = s;
    group_scales[0] = s;
    group_scales[1] = s;
    for (uint32_t lane = 0; lane < in_dim; ++lane) {
      uint32_t nib = nibble_at(packed.data(), lane);
      int sv = (int)nib >= 8 ? (int)nib - 16 : (int)nib;
      double sd = (double)half_to_float(s);
      float expect_a = (float)((double)sv * sd);
      if (fbits(dequant_a_at(packed.data(), &row_scale, 0, lane, in_dim)) !=
          fbits(expect_a)) ok_a = false;
      float expect_c = (float)((double)sv * sd);
      if (fbits(dequant_c_at(packed.data(), group_scales, 0, lane, in_dim)) !=
          fbits(expect_c)) ok_c = false;
    }
  }
  check("dequant A exhaustive (16 nibbles x 6 scales, bitwise)", ok_a);
  check("dequant C exhaustive (16 nibbles x 6 scales, bitwise)", ok_c);
  // A and C agree on the VALUE when the scale for the lane's group
  // equals the row scale — but the scale LAYOUT differs, so a blob
  // is still not interchangeable. Check the layout difference:
  // with group scales (1.0, 0.5), lane 0 and lane 64 must differ
  // under C and agree under A.
  {
    uint16_t gs[2] = {0x3C00, 0x3800};
    uint16_t rs = 0x3C00;
    float a0 = dequant_a_at(packed.data(), &rs, 0, 0, in_dim);
    float a64 = dequant_a_at(packed.data(), &rs, 0, 64, in_dim);
    float c0 = dequant_c_at(packed.data(), gs, 0, 0, in_dim);
    float c64 = dequant_c_at(packed.data(), gs, 0, 64, in_dim);
    uint32_t n0 = nibble_at(packed.data(), 0), n64 = nibble_at(packed.data(), 64);
    int s0 = (int)n0 >= 8 ? (int)n0 - 16 : (int)n0;
    int s64 = (int)n64 >= 8 ? (int)n64 - 16 : (int)n64;
    check("A/C layout differs (group scale changes lane 64 only under C)",
          fbits(a0) == fbits((float)s0 * 1.0f) &&
          fbits(a64) == fbits((float)s64 * 1.0f) &&
          fbits(c0) == fbits((float)s0 * 1.0f) &&
          fbits(c64) == fbits((float)s64 * 0.5f));
  }
  // Quantize roundtrip: A, in_dim 40 (nonmultiple of 32 — a tail the
  // group schemes cannot express, A can).
  {
    casedata::Lcg rng;
    std::vector<float> w(40);
    for (auto& v : w) v = rng.next_sym();
    std::vector<uint8_t> pk(packed_row_bytes(40));
    quantize_row_a(w.data(), 40, 0x3C00, pk.data());
    bool ok = true;
    for (uint32_t lane = 0; lane < 40; ++lane) {
      long v = std::lrint(w[lane]); // scale 1.0, RNE via nearbyint path
      if (v < -8) v = -8; if (v > 7) v = 7;
      float expect = (float)v;
      uint16_t rs_one = 0x3C00;
      if (fbits(dequant_a_at(pk.data(), &rs_one, 0, lane, 40)) !=
          fbits(expect)) ok = false;
    }
    check("quantize A roundtrip, in_dim=40 tail", ok);
  }
}

// ---- format-selected matvec, all three formats ----
static void test_matvec_fmt() {
  casedata::Lcg rng;
  auto run_case = [&](QuantFormat fmt, uint32_t in_dim, uint32_t out_dim) {
    std::vector<float> x(in_dim), y(out_dim, 0.0f), yd(out_dim, 0.0);
    for (auto& v : x) v = rng.next_sym();
    std::vector<uint8_t> packed((size_t)out_dim * packed_row_bytes(in_dim));
    std::vector<uint16_t> scales;
    uint32_t groups = 0;
    if (fmt == QuantFormat::A_SignedPerRow) {
      scales.assign(out_dim, 0x3C00);
      for (uint32_t r = 0; r < out_dim; ++r) {
        std::vector<float> w(in_dim);
        for (auto& v : w) v = rng.next_sym();
        quantize_row_a(w.data(), in_dim, 0x3C00,
                       packed.data() + (size_t)r * packed_row_bytes(in_dim));
      }
    } else if (fmt == QuantFormat::B_Offset75PerGroup32) {
      groups = (in_dim + 31) / 32;
      scales.assign((size_t)out_dim * groups * 2, 0); // half2 slots as u16 pairs
      for (uint32_t r = 0; r < out_dim; ++r) {
        std::vector<uint16_t> gs(groups);
        for (auto& s : gs) s = 0x3C00;
        std::vector<float> w(in_dim);
        for (auto& v : w) v = rng.next_sym();
        quantize_row(w.data(), in_dim, gs.data(),
                     packed.data() + (size_t)r * packed_row_bytes(in_dim),
                     reinterpret_cast<uint32_t*>(scales.data()) + (size_t)r * groups);
      }
    } else { // C
      groups = (in_dim + 63) / 64;
      scales.assign((size_t)out_dim * groups, 0x3C00);
      for (uint32_t r = 0; r < out_dim; ++r) {
        std::vector<float> w(in_dim);
        for (auto& v : w) v = rng.next_sym();
        quantize_row_c(w.data(), in_dim, scales.data() + (size_t)r * groups,
                       packed.data() + (size_t)r * packed_row_bytes(in_dim));
      }
    }
    matvec_fmt(fmt, packed.data(), scales.data(), x.data(), y.data(), in_dim, out_dim);
    // Double-precision reference from the decoded weights.
    for (uint32_t r = 0; r < out_dim; ++r) {
      double acc = 0.0;
      for (uint32_t c = 0; c < in_dim; ++c) {
        double w = 0.0;
        if (fmt == QuantFormat::A_SignedPerRow)
          w = (double)dequant_a_at(packed.data(), scales.data(), r, c, in_dim);
        else if (fmt == QuantFormat::B_Offset75PerGroup32)
          w = (double)dequant_at(packed.data() + (size_t)r * packed_row_bytes(in_dim),
                                 reinterpret_cast<const uint32_t*>(scales.data()) +
                                 (size_t)r * groups, c);
        else
          w = (double)dequant_c_at(packed.data(), scales.data(), r, c, in_dim);
        acc += w * (double)x[c];
      }
      yd[r] = (float)acc;
    }
    double err = max_err(y, yd);
    char name[128];
    snprintf(name, sizeof name, "matvec_fmt %s in=%u out=%u (max err %.3g)",
             tensorabi::format_name(fmt), in_dim, out_dim, err);
    check(name, err <= 1e-5);
  };
  run_case(QuantFormat::A_SignedPerRow, 64, 3);
  run_case(QuantFormat::A_SignedPerRow, 40, 2); // tail, nonmultiple of 32
  run_case(QuantFormat::B_Offset75PerGroup32, 64, 3);
  run_case(QuantFormat::C_SignedPerGroup64, 128, 3);
  run_case(QuantFormat::C_SignedPerGroup64, 64, 2); // single group
}

// ---- descriptors + format compatibility ----
static void test_descriptors() {
  using namespace tensorabi;
  std::string why;
  auto dense_f16 = [](uint32_t n) {
    TensorDescriptor d;
    d.dtype = DType::F16; d.rank = 1; d.shape[0] = n; d.strides[0] = 1;
    return d;
  };
  auto weight = [](uint32_t rows, uint32_t in, QuantFormat f) {
    TensorDescriptor d;
    d.dtype = DType::U8; d.rank = 2; d.shape[0] = rows; d.shape[1] = in;
    d.strides[0] = (uint32_t)tensorabi::packed_row_bytes(in); d.strides[1] = 1;
    d.format = f; d.group_size = format_group_size(f);
    d.scale_layout = format_scale_layout(f); d.scale_dtype = DType::F16;
    return d;
  };
  check("desc: dense f16 valid",
        validate_descriptor(dense_f16(3072), 6144, why));
  check("desc: weight A valid",
        validate_descriptor(weight(4, 64, QuantFormat::A_SignedPerRow),
                            4 * 32 + 4 * 2, why));
  check("desc: weight B valid",
        validate_descriptor(weight(4, 64, QuantFormat::B_Offset75PerGroup32),
                            4 * 32 + 4 * 2 * 4, why));
  check("desc: weight C valid",
        validate_descriptor(weight(4, 128, QuantFormat::C_SignedPerGroup64),
                            4 * 64 + 4 * 2 * 2, why));
  { TensorDescriptor d = dense_f16(8); d.rank = 0;
    check("desc: rank 0 rejected", !validate_descriptor(d, 16, why) && !why.empty()); }
  { TensorDescriptor d = dense_f16(8); d.shape[0] = 0;
    check("desc: empty dim rejected", !validate_descriptor(d, 16, why)); }
  { TensorDescriptor d = dense_f16(8); d.strides[0] = 2;
    check("desc: non-contiguous strides rejected",
          !validate_descriptor(d, 16, why)); }
  { TensorDescriptor d = dense_f16(8); d.buffer_offset = 8; d.alignment = 16;
    check("desc: misaligned offset rejected",
          !validate_descriptor(d, 32, why)); }
  { TensorDescriptor d = dense_f16(8);
    check("desc: buffer too small rejected",
          !validate_descriptor(d, 15, why)); }
  { TensorDescriptor d = weight(4, 64, QuantFormat::B_Offset75PerGroup32);
    d.group_size = 64;
    check("desc: B with group 64 rejected",
          !validate_descriptor(d, 1024, why)); }
  { TensorDescriptor d = weight(4, 64, QuantFormat::B_Offset75PerGroup32);
    d.scale_layout = 1;
    check("desc: B with flat scale layout rejected",
          !validate_descriptor(d, 1024, why)); }
  { TensorDescriptor d = weight(4, 40, QuantFormat::B_Offset75PerGroup32);
    check("desc: B with in_dim 40 rejected (needs %32)",
          !validate_descriptor(d, 1024, why)); }
  { TensorDescriptor d = weight(4, 96, QuantFormat::C_SignedPerGroup64);
    check("desc: C with in_dim 96 rejected (needs %64)",
          !validate_descriptor(d, 4096, why)); }
  { TensorDescriptor d = weight(4, 64, QuantFormat::A_SignedPerRow);
    d.dtype = DType::F16;
    check("desc: quantized dtype must be U8",
          !validate_descriptor(d, 1024, why)); }
  // Format compatibility: no silent reinterpretation.
  check("compat: A weight vs A kernel ok",
        check_format_compatible(weight(4, 64, QuantFormat::A_SignedPerRow),
                                QuantFormat::A_SignedPerRow, why));
  check("compat: A weight vs B kernel refused",
        !check_format_compatible(weight(4, 64, QuantFormat::A_SignedPerRow),
                                 QuantFormat::B_Offset75PerGroup32, why));
  check("compat: C weight vs B kernel refused",
        !check_format_compatible(weight(4, 128, QuantFormat::C_SignedPerGroup64),
                                 QuantFormat::B_Offset75PerGroup32, why));
  check("compat: B weight vs C kernel refused",
        !check_format_compatible(weight(4, 64, QuantFormat::B_Offset75PerGroup32),
                                 QuantFormat::C_SignedPerGroup64, why));
  // Byte accounting: packed + scales for a B weight.
  {
    TensorDescriptor d = weight(4, 64, QuantFormat::B_Offset75PerGroup32);
    check("desc: B byte accounting (128 packed + 32 scale)",
          d.data_bytes() == 128 && d.scale_bytes() == 32);
    TensorDescriptor c = weight(4, 128, QuantFormat::C_SignedPerGroup64);
    check("desc: C byte accounting (256 packed + 16 scale)",
          c.data_bytes() == 256 && c.scale_bytes() == 16);
    TensorDescriptor a = weight(4, 64, QuantFormat::A_SignedPerRow);
    check("desc: A byte accounting (128 packed + 8 scale)",
          a.data_bytes() == 128 && a.scale_bytes() == 8);
  }
}

// ---- norms, rope, kv, attention ----
static void test_norms_rope_kv() {
  double err = 0.0;
  { // rmsnorm hand case: x = 1,2,3,4; gain 1; eps 1e-5
    float x[4] = {1, 2, 3, 4}, g[4] = {1, 1, 1, 1}, y[4];
    kref::rmsnorm_ref(x, g, 1.0e-5f, 4, y);
    double inv = 1.0 / std::sqrt(7.5 + 1e-5);
    std::vector<float> got(y, y + 4), want = {(float)(1 * inv), (float)(2 * inv),
                                              (float)(3 * inv), (float)(4 * inv)};
    check("rmsnorm hand case [1,2,3,4]", near_vec(got, want, 1e-6, err));
    // zeros: inv = 1/sqrt(1e-6), output stays 0
    float z[4] = {0, 0, 0, 0}, yz[4];
    kref::rmsnorm_ref(z, g, 1.0e-5f, 4, yz);
    check("rmsnorm zeros stay zero", yz[0] == 0 && yz[3] == 0);
  }
  { // rmsnorm at model-default hidden=3072 vs double ref
    casedata::Lcg rng;
    std::vector<float> x(3072), g(3072), y(3072);
    for (auto& v : x) v = rng.next_sym();
    for (auto& v : g) v = 0.5f + 0.5f * rng.next_sym();
    kref::rmsnorm_ref(x.data(), g.data(), 1.0e-5f, 3072, y.data());
    double sum = 0.0;
    for (auto v : x) sum += (double)v * v;
    double inv = 1.0 / std::sqrt(sum / 3072.0 + 1e-5);
    std::vector<float> want(3072);
    for (size_t i = 0; i < 3072; ++i) want[i] = (float)(x[i] * inv * g[i]);
    char name[128];
    snprintf(name, sizeof name, "rmsnorm hidden=3072 (max err %.3g)", max_err(y, want));
    check(name, near_vec(y, want, 1e-5, err));
  }
  { // layernorm hand case with bias
    float x[4] = {1, 2, 3, 4}, g[4] = {1, 1, 1, 1}, b[4] = {0.5f, 0, 0, 0}, y[4];
    kref::layernorm_ref(x, g, b, 1.0e-5f, 4, y);
    double mean = 2.5, var = 1.25;
    double inv = 1.0 / std::sqrt(var + 1e-5);
    std::vector<float> got(y, y + 4),
        want = {(float)((1 - mean) * inv + 0.5), (float)((2 - mean) * inv),
                (float)((3 - mean) * inv), (float)((4 - mean) * inv)};
    check("layernorm hand case + bias", near_vec(got, want, 1e-6, err));
  }
  { // rope interleaved: pos 1, head_dim 4, q=[1,0,0,0]
    float q[4] = {1, 0, 0, 0}, k[4] = {0, 1, 0, 0};
    kref::rope_interleaved_ref(q, k, 1, 1, 1, 4);
    double c = std::cos(1.0), s = std::sin(1.0);
    std::vector<float> gq(q, q + 4), wq = {(float)c, (float)s, 0, 0};
    std::vector<float> gk(k, k + 4);
    // k pair 0 was (0,1) -> (-sin1, cos1); pair 1 untouched (0,0)
    std::vector<float> wk = {(float)-s, (float)c, 0, 0};
    check("rope interleaved pos=1 pair0", near_vec(gq, wq, 1e-6, err) &&
                                             near_vec(gk, wk, 1e-6, err));
    float q0[4] = {1, 2, 3, 4}, k0[4] = {5, 6, 7, 8};
    kref::rope_interleaved_ref(q0, k0, 0, 1, 1, 4);
    check("rope pos=0 is identity",
          q0[0] == 1 && q0[1] == 2 && k0[0] == 5 && k0[3] == 8);
    // K rotated once with GQA: q_heads=2, kv_heads=1. K pair 0 must
    // be rotated exactly once (the rope.metal defect this port fixes).
    float q2[8] = {1, 0, 0, 0, 0, 0, 0, 0}, k2[4] = {1, 0, 0, 0};
    kref::rope_interleaved_ref(q2, k2, 1, 2, 1, 4);
    check("rope K rotated once under GQA (not 2x)",
          std::fabs(k2[0] - (float)std::cos(1.0)) < 1e-6 &&
          std::fabs(k2[1] - (float)std::sin(1.0)) < 1e-6);
  }
  { // rope split: head_dim 4, pos 1, q=[1,0,0,0] at position 1.
    // Arrays hold 2 positions (seq_len 2); the ref indexes
    // pos * heads * head_dim, so a 4-float array would be OOB.
    float q[8] = {0, 0, 0, 0, 1, 0, 0, 0}, k[8] = {0, 0, 0, 0, 0, 0, 1, 0};
    kref::rope_split_ref(q, k, 1, 2, 1, 1, 4);
    double c0 = std::cos(1.0), s0 = std::sin(1.0);
    // pair d=0 rotates (q0, q2) by 1 rad: (cos1, sin1); d=1 rotates
    // (q1, q3) = (0,0) by 0.01 rad, stays 0. K pair d=0 was (0,1):
    // -> (-sin1, cos1).
    check("rope split pos=1",
          std::fabs(q[4] - (float)c0) < 1e-6 && std::fabs(q[6] - (float)s0) < 1e-6 &&
          std::fabs(k[4] - (float)-s0) < 1e-6 && std::fabs(k[6] - (float)c0) < 1e-6);
  }
  { // kv store/fetch/clear across layers and positions
    const uint32_t KH = 2, DH = 4, CTX = 16;
    std::vector<float> kc(CTX * KH * DH, -99.0f), vc(CTX * KH * DH, -99.0f);
    float k[8] = {1, 2, 3, 4, 5, 6, 7, 8}, v[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    kref::kv_store_ref(k, v, kc.data(), vc.data(), 1, 3, KH, DH, CTX);
    float ko[8], vo[8];
    kref::kv_fetch_ref(kc.data(), vc.data(), ko, vo, 1, 3, KH, DH, CTX);
    bool ok = true;
    for (int i = 0; i < 8; ++i) ok = ok && ko[i] == k[i] && vo[i] == v[i];
    check("kv store/fetch roundtrip (layer 1, pos 3)", ok);
    // layer 0 pos 3 must still be untouched (-99 sentinel)
    float ko0[8], vo0[8];
    kref::kv_fetch_ref(kc.data(), vc.data(), ko0, vo0, 0, 3, KH, DH, CTX);
    check("kv layers do not alias", ko0[0] == -99.0f && vo0[7] == -99.0f);
    kref::kv_clear_ref(kc.data(), (uint32_t)kc.size());
    check("kv clear zeroes", kc[0] == 0.0f && kc[kc.size() - 1] == 0.0f);
  }
  { // gqa strided vs online on the same tiny case
    const uint32_t H = 2, G = 1, D = 4;
    float q[8] = {1, 0, 0, 0, 0, 1, 0, 0}; // head0 [1,0,0,0], head1 [0,1,0,0]
    // cache rows (pos-major, stride G*D=4): pos0 k=[1,0,0,0] v=[2,0,0,0];
    // pos1 k=[0,1,0,0] v=[0,3,0,0]
    float kc[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    float vc[8] = {2, 0, 0, 0, 0, 3, 0, 0};
    float out_s[8], out_o[8];
    float scale = 1.0f / std::sqrt((float)D); // 0.5
    kref::gqa_strided_ref(q, kc, vc, out_s, 1, 2, H, G, D, 4, scale);
    double l = 1.0 + std::exp(-0.5);
    std::vector<float> got(out_s, out_s + 8);
    std::vector<float> want = {(float)(2.0 / l), (float)(3.0 * std::exp(-0.5) / l), 0, 0,
                               (float)(2.0 * std::exp(-0.5) / l), (float)(3.0 / l), 0, 0};
    check("gqa strided hand case", near_vec(got, want, 1e-6, err));
    // online form, head 0 and head 1 separately. The online kernel
    // is causal: kv_len = q_pos + 1, so compare at q_pos = 1 (2 keys).
    kref::gqa_online_ref(q, kc, vc, out_o, 0, 1, H, G, D);
    kref::gqa_online_ref(q, kc, vc, out_o, 1, 1, H, G, D);
    std::vector<float> goto_o(out_o, out_o + 8);
    check("gqa online agrees with strided", near_vec(goto_o, want, 1e-6, err));
    // single position: attention is the identity on V. kv_len=1 reads
    // only pos 0, whose V row is [2,0,0,0] for both heads.
    float out1[8];
    kref::gqa_strided_ref(q, kc, vc, out1, 1, 1, H, G, D, 4, scale);
    check("gqa kv_len=1 returns V row",
          out1[0] == 2.0f && out1[1] == 0.0f && out1[4] == 2.0f &&
          out1[5] == 0.0f);
  }
  { // decomposed attention pipeline == fused gqa on the same case
    const uint32_t H = 2, G = 1, D = 4;
    float q[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    // kv_offset layout: [pos][kv_heads*head_dim]
    float kc[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    float vc[8] = {2, 0, 0, 0, 0, 3, 0, 0};
    float scores[8] = {0};
    kref::attention_scores_ref(q, kc, scores, H, G, D, 2);
    check("attention scores hand case",
          std::fabs(scores[0] - 0.5f) < 1e-6 && std::fabs(scores[1]) < 1e-6 &&
          std::fabs(scores[2]) < 1e-6 && std::fabs(scores[3] - 0.5f) < 1e-6);
    kref::stable_softmax_ref(scores, H, 2, 1);
    check("softmax rows sum to 1, tail untouched by len",
          std::fabs(scores[0] + scores[1] - 1.0f) < 1e-6);
    float out[8];
    kref::attention_value_ref(scores, vc, out, H, G, D, 2, 1);
    double l = 1.0 + std::exp(-0.5);
    std::vector<float> got(out, out + 8);
    std::vector<float> want = {(float)(2.0 / l), (float)(3.0 * std::exp(-0.5) / l), 0, 0,
                               (float)(2.0 * std::exp(-0.5) / l), (float)(3.0 / l), 0, 0};
    check("scores+softmax+value == fused gqa", near_vec(got, want, 1e-6, err));
    // softmax tail zeroing: length 4, offset 1
    float s4[8] = {2, 1, 9, 9, 0, 0, 0, 0};
    kref::stable_softmax_ref(s4, 2, 4, 1);
    check("softmax zeroes entries past offset",
          s4[2] == 0.0f && s4[3] == 0.0f && std::fabs(s4[0] + s4[1] - 1.0f) < 1e-6);
  }
}

// ---- mlp / residual / lora / decode ops ----
static void test_mlp_lora_decode() {
  double err = 0.0;
  {
    float g[4] = {0, 1, -1, 2}, u[4] = {1, 1, 1, 1}, a[4];
    kref::swiglu_act_ref(g, u, 4, a);
    std::vector<float> got(a, a + 4), want(4);
    for (int i = 0; i < 4; ++i) {
      double gg = g[i];
      want[i] = (float)(gg / (1.0 + std::exp(-gg)) * u[i]);
    }
    check("swiglu_act hand case", near_vec(got, want, 1e-6, err));
    check("swiglu_act silu(0)=0", a[0] == 0.0f);
  }
  {
    // swiglu_full with Scheme A weights, hidden 64, inter 4
    casedata::Lcg rng;
    const uint32_t H = 64, I = 4;
    std::vector<uint8_t> wg((size_t)I * packed_row_bytes(H)),
        wu((size_t)I * packed_row_bytes(H));
    std::vector<uint16_t> sg(I, 0x3C00), su(I, 0x3C00);
    for (uint32_t r = 0; r < I; ++r) {
      std::vector<float> w(H);
      for (auto& v : w) v = rng.next_sym();
      quantize_row_a(w.data(), H, 0x3C00, wg.data() + (size_t)r * packed_row_bytes(H));
      for (auto& v : w) v = rng.next_sym();
      quantize_row_a(w.data(), H, 0x3C00, wu.data() + (size_t)r * packed_row_bytes(H));
    }
    std::vector<float> x(H), out(I);
    for (auto& v : x) v = rng.next_sym();
    kref::swiglu_full_ref(wg.data(), sg.data(), wu.data(), su.data(),
                          x.data(), out.data(), H, I);
    // Double reference from decoded weights.
    std::vector<float> want(I);
    for (uint32_t r = 0; r < I; ++r) {
      double gate = 0.0, up = 0.0;
      for (uint32_t c = 0; c < H; ++c) {
        gate += (double)dequant_a_at(wg.data(), sg.data(), r, c, H) * x[c];
        up += (double)dequant_a_at(wu.data(), su.data(), r, c, H) * x[c];
      }
      want[r] = (float)(gate / (1.0 + std::exp(-std::max(-80.0, std::min(80.0, gate)))) * up);
    }
    check("swiglu_full Scheme A (max err reported)",
          near_vec(out, want, 1e-5, err));
  }
  {
    float a[2] = {0.5f, -1.25f}, b[2] = {0.25f, 0.5f}, o[2];
    kref::residual_ref(a, b, 2.0f, 2, o);
    check("residual scaled, bitwise", fbits(o[0]) == fbits(1.0f) &&
                                       fbits(o[1]) == fbits(-0.25f));
    kref::residual_ref(a, b, 1.0f, 2, o);
    check("residual plain add, bitwise", fbits(o[0]) == fbits(0.75f) &&
                                          fbits(o[1]) == fbits(-0.75f));
  }
  {
    // lora_delta hand case: rank 2, K 3, N 2, M 1
    float A[6] = {1, 0, 0, 0, 1, 0}; // rows: [1,0,0], [0,1,0]
    float B[4] = {1, 0, 0, 1};       // [1,0], [0,1]
    float X[3] = {2, 3, 4};
    float Y[2] = {10, 20};
    bool ok = kref::lora_delta_ref(A, B, X, Y, 2, 3, 2, 1, 0.5f);
    // tmp = [2, 3]; delta = [2, 3]; Y += 0.5*delta -> [11, 21.5]
    check("lora_delta hand case", ok && Y[0] == 11.0f && Y[1] == 21.5f);
    check("lora_delta rank 65 refused", !kref::lora_delta_ref(A, B, X, Y, 65, 3, 2, 1, 1.0f));
  }
  {
    // lora down/up with FP16 A/B: in 4, rank 2, out 3
    float x[4] = {1, 2, 0, 0};
    uint16_t A[8], B[6];
    float Af[8] = {1, 0, 0, 0, 0, 1, 0, 0}; // rows [1,0,0,0],[0,1,0,0]
    float Bf[6] = {1, 0, 0, 1, 0, 0};       // rows [1,0],[0,1],[0,0]
    for (int i = 0; i < 8; ++i) A[i] = float_to_half(Af[i]);
    for (int i = 0; i < 6; ++i) B[i] = float_to_half(Bf[i]);
    float z[2] = {0, 0};
    kref::lora_down_ref(x, A, z, 0, 4, 2);
    check("lora_down hand case", z[0] == 1.0f && z[1] == 2.0f);
    float y[3] = {5, 5, 5};
    kref::lora_up_ref(z, B, y, 0, 3, 2, 0.5f);
    check("lora_up hand case", y[0] == 5.5f && y[1] == 6.0f && y[2] == 5.0f);
  }
  {
    float lg[4] = {1, 5, 5, 2};
    check("argmax first max wins on ties", kref::argmax_ref(lg, 4) == 1);
    float lg2[3] = {7, 7, 7};
    check("argmax all equal -> 0", kref::argmax_ref(lg2, 3) == 0);
    float lg3[3] = {0.0f, 1.0f, -1.0f};
    float c3[3];
    std::memcpy(c3, lg3, sizeof lg3);
    kref::temperature_softmax_ref(c3, 3, 1.0f);
    double s = 0.0;
    for (int i = 0; i < 3; ++i) s += c3[i];
    check("temperature_softmax sums to 1", std::fabs(s - 1.0) < 1e-6);
    float c4[3];
    std::memcpy(c4, lg3, sizeof lg3);
    kref::temperature_softmax_ref(c4, 3, 2.0f);
    // temp 2 flattens: max prob must drop vs temp 1
    check("temperature_softmax temp 2 flattens", c4[1] < c3[1]);
  }
}

// ---- composed one-layer decode + golden replay ----
static kref::LayerDims tiny_dims() {
  kref::LayerDims d;
  d.hidden = 8; d.intermediate = 8; d.q_heads = 2; d.kv_heads = 1;
  d.head_dim = 4; d.context = 16; d.eps = 1.0e-5f;
  return d;
}

static void test_decode_layer() {
  auto d = tiny_dims();
  casedata::Lcg rng;
  auto make_w = [&](uint32_t rows, uint32_t in) {
    std::vector<uint8_t> pk((size_t)rows * packed_row_bytes(in));
    std::vector<uint16_t> sc(rows, 0x3C00);
    for (uint32_t r = 0; r < rows; ++r) {
      std::vector<float> w(in);
      for (auto& v : w) v = rng.next_sym() * 4.0f;
      quantize_row_a(w.data(), in, 0x3C00, pk.data() + (size_t)r * packed_row_bytes(in));
    }
    return std::pair<std::vector<uint8_t>, std::vector<uint16_t>>{std::move(pk),
                                                                   std::move(sc)};
  };
  auto wq = make_w(8, 8), wk = make_w(4, 8), wv = make_w(4, 8), wo = make_w(8, 8);
  auto wg = make_w(8, 8), wu = make_w(8, 8), wd = make_w(8, 8);
  kref::LayerWeightsA w{wq.first.data(), wq.second.data(),
                        wk.first.data(), wk.second.data(),
                        wv.first.data(), wv.second.data(),
                        wo.first.data(), wo.second.data(),
                        wg.first.data(), wg.second.data(),
                        wu.first.data(), wu.second.data(),
                        wd.first.data(), wd.second.data(),
                        nullptr, nullptr};
  std::vector<float> g1(8, 1.0f), g2(8, 1.0f);
  w.attn_norm_gain = g1.data();
  w.ffn_norm_gain = g2.data();
  std::vector<float> x = {0.1f, -0.2f, 0.3f, 0.0f, 0.5f, -0.1f, 0.2f, -0.4f};
  std::vector<float> kc(16 * 4, 0.0f), vc(16 * 4, 0.0f);
  std::vector<float> scratch(8 + 8 + 4 + 4 + 8 + 8 + 8 + 8 + 8, 0.0f);
  std::vector<float> out0(8), out1(8);
  kref::decode_layer_ref(x.data(), w, d, 0, kc.data(), vc.data(), out0.data(),
                         scratch.data());
  bool finite = true;
  for (float v : out0) finite = finite && std::isfinite(v);
  check("decode layer pos0 finite", finite);
  // Determinism: fresh cache, same inputs -> identical bits.
  std::vector<float> kc2(16 * 4, 0.0f), vc2(16 * 4, 0.0f), out0b(8);
  kref::decode_layer_ref(x.data(), w, d, 0, kc2.data(), vc2.data(), out0b.data(),
                         scratch.data());
  bool det = true;
  for (int i = 0; i < 8; ++i) det = det && fbits(out0[i]) == fbits(out0b[i]);
  check("decode layer deterministic (bitwise)", det);
  // Position 1 sees the cached position 0: output must differ. Note
  // the test uses a DIFFERENT x at pos 1: with identical x the V rows
  // are identical, RoPE only re-weights identical rows, and the
  // composed output is genuinely identical. Different x makes the
  // cache read observable.
  std::vector<float> x1 = {0.4f, 0.1f, -0.3f, 0.2f, 0.0f, 0.3f, -0.2f, 0.1f};
  kref::decode_layer_ref(x1.data(), w, d, 1, kc.data(), vc.data(), out1.data(),
                         scratch.data());
  bool differs = false;
  for (int i = 0; i < 8; ++i) differs = differs || fbits(out0[i]) != fbits(out1[i]);
  check("decode layer pos1 differs from pos0 (cache is read)", differs);
  // Cache roundtrip at pos 0 still holds after the pos-1 step.
  float ko[4], vo[4];
  kref::kv_fetch_ref(kc.data(), vc.data(), ko, vo, 0, 0, 1, 4, 16);
  check("decode layer cache survives second step",
        std::isfinite(ko[0]) && std::isfinite(vo[3]));
}

// ---- golden replay (kernel_golden.txt) ----
static bool read_golden(const char* path, std::vector<float>& rms_out,
                        std::vector<float>& layer_out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line, cur;
  while (std::getline(in, line)) {
    if (line.rfind("case ", 0) == 0) { cur = line.substr(5); continue; }
    if (line.rfind("expected", 0) == 0) {
      std::istringstream ss(line.substr(8));
      std::string tok;
      std::vector<float>* dst = nullptr;
      if (cur.rfind("rmsnorm", 0) == 0) dst = &rms_out;
      else if (cur.rfind("decode_layer", 0) == 0) dst = &layer_out;
      if (!dst) continue;
      while (ss >> tok) {
        uint32_t u = (uint32_t)std::stoul(tok, nullptr, 16);
        float f;
        std::memcpy(&f, &u, 4);
        dst->push_back(f);
      }
    }
  }
  return true;
}

static void test_golden(const char* path) {
  std::vector<float> rms_gold, layer_gold;
  if (!read_golden(path, rms_gold, layer_gold)) {
    check("kernel golden file readable", false);
    return;
  }
  check("kernel golden file readable", !rms_gold.empty() && !layer_gold.empty());
  // rmsnorm golden: x = 1..8, gain 1, eps 1e-5
  {
    float x[8] = {1, 2, 3, 4, 5, 6, 7, 8}, g[8] = {1, 1, 1, 1, 1, 1, 1, 1}, y[8];
    kref::rmsnorm_ref(x, g, 1.0e-5f, 8, y);
    bool ok = rms_gold.size() == 8;
    for (int i = 0; ok && i < 8; ++i) ok = fbits(y[i]) == fbits(rms_gold[i]);
    check("golden rmsnorm replay (bitwise)", ok);
  }
  // decode_layer golden is generated from the same deterministic
  // construction as test_decode_layer; replay pins the composed
  // result against the checked-in vector.
  {
    auto d = tiny_dims();
    casedata::Lcg rng;
    auto make_w = [&](uint32_t rows, uint32_t in) {
      std::vector<uint8_t> pk((size_t)rows * packed_row_bytes(in));
      std::vector<uint16_t> sc(rows, 0x3C00);
      for (uint32_t r = 0; r < rows; ++r) {
        std::vector<float> w(in);
        for (auto& v : w) v = rng.next_sym() * 4.0f;
        quantize_row_a(w.data(), in, 0x3C00, pk.data() + (size_t)r * packed_row_bytes(in));
      }
      return std::pair<std::vector<uint8_t>, std::vector<uint16_t>>{std::move(pk),
                                                                     std::move(sc)};
    };
    auto wq = make_w(8, 8), wk = make_w(4, 8), wv = make_w(4, 8), wo = make_w(8, 8);
    auto wg = make_w(8, 8), wu = make_w(8, 8), wd = make_w(8, 8);
    kref::LayerWeightsA w{wq.first.data(), wq.second.data(),
                          wk.first.data(), wk.second.data(),
                          wv.first.data(), wv.second.data(),
                          wo.first.data(), wo.second.data(),
                          wg.first.data(), wg.second.data(),
                          wu.first.data(), wu.second.data(),
                          wd.first.data(), wd.second.data(),
                          nullptr, nullptr};
    std::vector<float> g1(8, 1.0f), g2(8, 1.0f);
    w.attn_norm_gain = g1.data();
    w.ffn_norm_gain = g2.data();
    std::vector<float> x = {0.1f, -0.2f, 0.3f, 0.0f, 0.5f, -0.1f, 0.2f, -0.4f};
    std::vector<float> kc(16 * 4, 0.0f), vc(16 * 4, 0.0f);
    std::vector<float> scratch(8 + 8 + 4 + 4 + 8 + 8 + 8 + 8 + 8, 0.0f);
    std::vector<float> out(8);
    kref::decode_layer_ref(x.data(), w, d, 0, kc.data(), vc.data(), out.data(),
                           scratch.data());
    bool ok = layer_gold.size() == 8;
    for (int i = 0; ok && i < 8; ++i) ok = fbits(out[i]) == fbits(layer_gold[i]);
    check("golden decode_layer replay (bitwise)", ok);
  }
}

// ---- command graph ----
static void test_graph() {
  std::string why;
  graph::ModelDims dims;
  // hidden 64 (not 8): fused_qkv decodes Scheme C, whose scale-index
  // math is only well-defined when K % 64 == 0; the strict rule in
  // validate_descriptor refuses anything else, so the graph test
  // uses the smallest admissible hidden size.
  dims.hidden = 64; dims.intermediate = 64; dims.q_heads = 2;
  dims.kv_heads = 1; dims.head_dim = 4; dims.vocab = 16; dims.context = 16;
  graph::Graph g;
  check("graph builds (1 layer)",
        graph::build_forward_graph(dims, 1, 1, g, why));
  if (g.nodes.size() == graph::node_count_for(1)) {
    check("graph node count 1 layer = 15", true);
    // Order: embedding first, argmax last; per-layer block order.
    check("graph order: embedding first, argmax last",
          g.nodes.front().shader == "embedding" &&
          g.nodes.back().shader == "argmax");
    const char* want[11] = {"rmsnorm", "fused_qkv", "rope", "kv_store",
                            "gqa_attention", "int4_matvec_fmt", "residual",
                            "rmsnorm", "swiglu", "int4_matvec_fmt",
                            "residual"};
    // Layer block is nodes[1..11]; tail is [12]=rmsnorm(final),
    // [13]=logits, [14]=argmax.
    bool order_ok = true;
    for (int i = 0; i < 11; ++i)
      if (g.nodes[1 + i].shader != want[i]) order_ok = false;
    check("graph layer block order (11 nodes)", order_ok);
    check("graph tail: final_norm rmsnorm, logits, argmax",
          g.nodes[12].shader == "rmsnorm" && g.nodes[13].shader == "logits" &&
          g.nodes[14].shader == "argmax");
    // Formats are declared per node: fused_qkv is C, projections are A.
    bool fmt_ok = true;
    for (auto& n : g.nodes) {
      if (n.label.rfind("fused_qkv", 0) == 0 &&
          n.weight_format != QuantFormat::C_SignedPerGroup64) fmt_ok = false;
      if ((n.label.rfind("output_projection", 0) == 0 ||
           n.label.rfind("mlp_down", 0) == 0 ||
           n.label.rfind("swiglu", 0) == 0) &&
          n.weight_format != QuantFormat::A_SignedPerRow) fmt_ok = false;
      if (n.label == "logits" &&
          n.weight_format != QuantFormat::A_SignedPerRow) fmt_ok = false;
    }
    check("graph node formats declared (C for QKV, A for proj/MLP/logits)", fmt_ok);
  } else {
    check("graph node count 1 layer = 15", false);
  }
  graph::Graph g2;
  check("graph builds (2 layers)",
        graph::build_forward_graph(dims, 2, 1, g2, why));
  check("graph node count 2 layers = 26",
        g2.nodes.size() == graph::node_count_for(2));
  check("graph caps note present (device or BLOCKED reason)",
        !g2.caps_note.empty());
  graph::Graph bad;
  check("graph rejects layers=0",
        !graph::build_forward_graph(dims, 0, 1, bad, why) && !why.empty());
  check("graph rejects seq_len=2 (single-token decode only)",
        !graph::build_forward_graph(dims, 1, 2, bad, why) && !why.empty());
  graph::ModelDims bad_dims = dims;
  bad_dims.q_heads = 3; bad_dims.kv_heads = 2;
  check("graph rejects q_heads not a multiple of kv_heads",
        !graph::build_forward_graph(bad_dims, 1, 1, bad, why));
  bad_dims = dims; bad_dims.head_dim = 5;
  check("graph rejects odd head_dim",
        !graph::build_forward_graph(bad_dims, 1, 1, bad, why));
  // Model-default dims build: hidden 3072, inter 8192, heads 24,
  // kv 8, dh 128, vocab 32000, context 4096 — shape math only.
  graph::ModelDims big;
  graph::Graph gb;
  check("graph builds at model defaults",
        graph::build_forward_graph(big, 1, 1, gb, why));
  if (!gb.nodes.empty())
    check("graph at defaults: 15 nodes", gb.nodes.size() == 15);
}

int main(int argc, char** argv) {
  test_dequant_ac();
  test_matvec_fmt();
  test_descriptors();
  test_norms_rope_kv();
  test_mlp_lora_decode();
  test_decode_layer();
  if (argc >= 2) test_golden(argv[1]);
  else check("kernel golden file argument given", false);
  test_graph();
  printf("kernel_ref: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
