// test_int4_oracle.cpp — CPU oracle verification.
// Bitwise-exact tests (packing, dequantization, FP16 conversion)
// are kept separate from tolerance tests (matvec accumulation),
// because only the latter involves summation order.
// Usage: test_int4_oracle <path-to-int4_golden.txt>
#include "int4_abi.h"
#include "case_data.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
#define CHECK(name, cond) do { \
  if (cond) { ++g_pass; printf("PASS %s\n", name); } \
  else { ++g_fail; printf("FAIL %s\n", name); } } while (0)

static uint32_t fbits(float v) { uint32_t u; std::memcpy(&u, &v, 4); return u; }
using namespace int4abi;

static void test_fp16_constants() {
  CHECK("fp16 1.0", float_to_half(1.0f) == 0x3C00);
  CHECK("fp16 -2.0", float_to_half(-2.0f) == 0xC000);
  CHECK("fp16 0.1", float_to_half(0.1f) == 0x2E66);
  CHECK("fp16 max", float_to_half(65504.0f) == 0x7BFF);
  CHECK("fp16 65520->inf (RNE)", float_to_half(65520.0f) == 0x7C00);
  CHECK("fp16 +0", float_to_half(0.0f) == 0x0000);
  CHECK("fp16 -0", float_to_half(-0.0f) == 0x8000);
  CHECK("fp16 min subnormal", float_to_half(5.960464477539063e-08f) == 0x0001); // 2^-24
  CHECK("fp16 2^-25 tie->0", float_to_half(2.980232238769531e-08f) == 0x0000);
  CHECK("fp16 3*2^-25 tie->2", float_to_half(8.940696716308594e-08f) == 0x0002);
  CHECK("fp16 back 0.1", half_to_float(0x2E66) == 0.0999755859375f);
  CHECK("fp16 back max", half_to_float(0x7BFF) == 65504.0f);
}

static void test_fp16_roundtrip_exhaustive() {
  bool ok = true;
  for (uint32_t h = 0; h < 65536; ++h) {
    float f = half_to_float((uint16_t)h);
    uint16_t h2 = float_to_half(f);
    if (std::isnan(f)) {
      // NaN: class must survive; payload is canonicalized.
      if (!(((h2 >> 10) & 0x1F) == 0x1F && (h2 & 0x3FF))) { ok = false; break; }
    } else if (h2 != h) { ok = false; break; }
  }
  CHECK("fp16 exhaustive roundtrip (65536 patterns)", ok);
}

static void test_dequant_exhaustive() {
  const uint16_t scales[] = {0x0000, 0x0001, 0x3555, 0x3800, 0x3C00, 0x7BFF, 0xBC00};
  bool ok = true;
  for (uint16_t s : scales) {
    double sd = (double)half_to_float(s);
    for (uint32_t nib = 0; nib < 16; ++nib) {
      float expected = (float)(((double)nib - 7.5) * sd);
      float got = dequant_nibble(nib, s);
      if (fbits(got) != fbits(expected)) { ok = false; }
    }
  }
  CHECK("dequant exhaustive: 16 nibbles x 7 scales, bitwise", ok);
  CHECK("dequant nib 0 = -7.5*s", dequant_nibble(0, 0x3C00) == -7.5f);
  CHECK("dequant nib 15 = +7.5*s", dequant_nibble(15, 0x3C00) == 7.5f);
  CHECK("dequant nib 7 = -0.5*s", dequant_nibble(7, 0x3C00) == -0.5f);
  CHECK("dequant nib 8 = +0.5*s", dequant_nibble(8, 0x3C00) == 0.5f);
}

static void test_packing() {
  CHECK("geometry 32 -> 16 bytes", packed_row_bytes(32) == 16);
  CHECK("geometry 33 -> 17 bytes", packed_row_bytes(33) == 17);
  CHECK("geometry 64 -> 2 slots", scale_slots_per_row(64) == 2);
  uint8_t b[1] = {0xAB};
  CHECK("nibble order: lane0 = low", nibble_at(b, 0) == 0xB);
  CHECK("nibble order: lane1 = high", nibble_at(b, 1) == 0xA);
  // quantize -> dequant exactness on representable values
  bool ok = true;
  for (uint32_t k = 0; k < 16; ++k) {
    float w[32] = {};
    w[0] = ((float)k - 7.5f) * 0.5f;
    uint16_t gs[1] = {0x3800}; // 0.5
    uint8_t packed[16]; uint32_t slots[1];
    quantize_row(w, 32, gs, packed, slots);
    if (nibble_at(packed, 0) != k) ok = false;
    if (dequant_at(packed, slots, 0) != w[0]) ok = false;
    if ((slots[0] & 0xFFFFu) != 0x3800 || (slots[0] >> 16) != 0) ok = false;
  }
  CHECK("quantize/dequant roundtrip exact, half2 slot layout", ok);
  // saturation
  float big[32] = {}; big[0] = 100.0f; big[1] = -100.0f;
  uint16_t one[1] = {0x3C00}; uint8_t pk[16]; uint32_t sl[1];
  quantize_row(big, 32, one, pk, sl);
  CHECK("quantize saturates high", nibble_at(pk, 0) == 15);
  CHECK("quantize saturates low", nibble_at(pk, 1) == 0);
}

static void test_tail_group() {
  const uint32_t in_dim = 40; // one full group + tail of 8
  float w[40]; for (int i = 0; i < 40; ++i) w[i] = 0.25f * (i - 20);
  uint16_t gs[2] = {0x3C00, 0x3800};
  uint8_t packed[20]; uint32_t slots[2];
  quantize_row(w, in_dim, gs, packed, slots);
  CHECK("tail: row is exactly 20 bytes", packed_row_bytes(in_dim) == 20);
  float x[40]; for (int i = 0; i < 40; ++i) x[i] = 0.1f * ((i % 7) - 3);
  float y[1];
  matvec(packed, slots, x, y, in_dim, 1);
  double ref = 0.0;
  for (uint32_t c = 0; c < in_dim; ++c) ref += (double)dequant_at(packed, slots, c) * (double)x[c];
  CHECK("tail: matvec matches double reference", std::fabs((double)y[0] - ref) < 1e-6);
}

static std::vector<std::string> tokens(std::ifstream& in) {
  std::vector<std::string> t; std::string s;
  while (in >> s) { if (s[0] == '#') { std::string line; std::getline(in, line); continue; } t.push_back(s); }
  return t;
}
static uint32_t hex32(const std::string& s) { return (uint32_t)std::stoul(s, nullptr, 16); }
static std::vector<uint8_t> hexbytes(const std::string& s) {
  std::vector<uint8_t> out;
  for (size_t i = 0; i + 1 < s.size(); i += 2) out.push_back((uint8_t)std::stoul(s.substr(i, 2), nullptr, 16));
  return out;
}

static void test_golden(const char* path) {
  std::ifstream in(path);
  if (!in) { CHECK("golden file opens", false); return; }
  auto t = tokens(in);
  size_t i = 0;
  auto next = [&]() { return t.at(i++); };
  // case dequant_row
  if (!(next() == "case" && next() == "dequant_row")) { CHECK("golden dequant case header", false); return; }
  next(); // in_dim=64
  next(); // "scales"
  uint32_t s0 = hex32(next()), s1 = hex32(next());
  next(); // "packed"
  auto packed = hexbytes(next());
  next(); // "expected"
  uint32_t slots[2] = {s0, s1};
  bool ok = packed.size() == 32;
  for (uint32_t lane = 0; lane < 64 && ok; ++lane) {
    uint32_t exp = hex32(next());
    if (fbits(dequant_at(packed.data(), slots, lane)) != exp) ok = false;
  }
  CHECK("golden dequant_row: 64 lanes bitwise", ok);
  // case matvec
  if (!(next() == "case" && next() == "matvec")) { CHECK("golden matvec case header", false); return; }
  next(); next(); // in_dim=64 out_dim=4
  next(); auto pk = hexbytes(next());
  next(); std::vector<uint32_t> sc; for (int k = 0; k < 8; ++k) sc.push_back(hex32(next()));
  next(); std::vector<float> x; for (int k = 0; k < 64; ++k) { uint32_t u = hex32(next()); float f; std::memcpy(&f, &u, 4); x.push_back(f); }
  next(); std::vector<uint32_t> exp;
  for (int k = 0; k < 4; ++k) exp.push_back(hex32(next()));
  std::vector<float> y(4);
  matvec(pk.data(), sc.data(), x.data(), y.data(), 64, 4);
  bool ok2 = true;
  for (int k = 0; k < 4; ++k) {
    float e; uint32_t u = exp[k]; std::memcpy(&e, &u, 4);
    if (std::fabs(y[k] - e) > 5e-7f) ok2 = false; // summation-order rounding only
  }
  CHECK("golden matvec: 4 outputs within 5e-7", ok2);
}

static void test_matvec_reference() {
  auto c = casedata::build_matvec_case();
  std::vector<float> y(c.out_dim);
  matvec(c.packed.data(), c.scales.data(), c.x.data(), y.data(), c.in_dim, c.out_dim);
  bool ok = true;
  for (uint32_t r = 0; r < c.out_dim; ++r)
    if (std::fabs((double)y[r] - c.expected_y[r]) > 1e-6) ok = false;
  CHECK("matvec vs double reference (64x4)", ok);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: test_int4_oracle <golden>\n"); return 1; }
  test_fp16_constants();
  test_fp16_roundtrip_exhaustive();
  test_dequant_exhaustive();
  test_packing();
  test_tail_group();
  test_matvec_reference();
  test_golden(argv[1]);
  printf("%d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
