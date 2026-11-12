// kernel_ref.cpp — see kernel_ref.h. Serial order matches the Metal
// sources; FP32 throughout. FP16 output rounding is the caller's job.
#include "kernel_ref.h"
#include <cstring>
#include <vector>

namespace kref {

void rmsnorm_ref(const float* x, const float* gain, float eps,
                 uint32_t n, float* y) {
  double sum = 0.0;
  for (uint32_t i = 0; i < n; ++i) sum += (double)x[i] * (double)x[i];
  float inv = safe_rsqrt((float)(sum / (double)n) + eps);
  for (uint32_t i = 0; i < n; ++i) y[i] = x[i] * inv * gain[i];
}

void layernorm_ref(const float* x, const float* gain, const float* bias,
                   float eps, uint32_t n, float* y) {
  double mean = 0.0, sq = 0.0;
  for (uint32_t i = 0; i < n; ++i) {
    mean += (double)x[i];
    sq += (double)x[i] * (double)x[i];
  }
  mean /= (double)n;
  double var = sq / (double)n - mean * mean;
  if (var < 0.0) var = 0.0;
  float inv = safe_rsqrt((float)var + eps);
  for (uint32_t i = 0; i < n; ++i) {
    float b = bias ? bias[i] : 0.0f;
    y[i] = (x[i] - (float)mean) * inv * gain[i] + b;
  }
}

void rope_interleaved_ref(float* q, float* k, uint32_t pos,
                          uint32_t q_heads, uint32_t kv_heads,
                          uint32_t head_dim) {
  for (uint32_t h = 0; h < q_heads; ++h) {
    for (uint32_t pair = 0; pair < head_dim / 2; ++pair) {
      uint32_t even = pair * 2, odd = even + 1;
      float angle = (float)pos *
                    std::pow(10000.0f, -((float)(2u * pair)) / (float)head_dim);
      float c = std::cos(angle), s = std::sin(angle);
      float q0 = q[h * head_dim + even], q1 = q[h * head_dim + odd];
      q[h * head_dim + even] = q0 * c - q1 * s;
      q[h * head_dim + odd] = q0 * s + q1 * c;
      // K is rotated ONCE per KV head (h < kv_heads). The literal
      // rope.metal text rotates the K pair inside this per-Q-head
      // loop, which re-rotates each K pair q_heads/kv_heads times
      // when q_heads > kv_heads; decode.metal rotates once, and
      // that is the semantics ported here and in rope.comp.
      if (h < kv_heads) {
        float k0 = k[h * head_dim + even], k1 = k[h * head_dim + odd];
        k[h * head_dim + even] = k0 * c - k1 * s;
        k[h * head_dim + odd] = k0 * s + k1 * c;
      }
    }
  }
}

void rope_split_ref(float* q, float* k, uint32_t pos, uint32_t seq_len,
                    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                    float theta) {
  (void)seq_len; // per-position call; seq_len is a dispatch bound, not math
  uint32_t half = head_dim / 2;
  for (uint32_t h = 0; h < q_heads; ++h) {
    for (uint32_t d = 0; d < half; ++d) {
      float freq = std::pow(theta, -2.0f * (float)d / (float)head_dim);
      float ang = (float)pos * freq;
      float c = std::cos(ang), s = std::sin(ang);
      size_t base = (size_t)pos * q_heads * head_dim + (size_t)h * head_dim;
      float q0 = q[base + d], q1 = q[base + d + half];
      q[base + d] = q0 * c - q1 * s;
      q[base + d + half] = q0 * s + q1 * c;
      if (h < kv_heads) {
        size_t kb = (size_t)pos * kv_heads * head_dim + (size_t)h * head_dim;
        float k0 = k[kb + d], k1 = k[kb + d + half];
        k[kb + d] = k0 * c - k1 * s;
        k[kb + d + half] = k0 * s + k1 * c;
      }
    }
  }
}

void kv_store_ref(const float* k, const float* v, float* k_cache,
                  float* v_cache, uint32_t layer, uint32_t pos,
                  uint32_t kv_heads, uint32_t head_dim, uint32_t context) {
  uint32_t layer_stride = context * kv_heads * head_dim;
  uint32_t base = layer * layer_stride;
  for (uint32_t h = 0; h < kv_heads; ++h)
    for (uint32_t d = 0; d < head_dim; ++d) {
      uint32_t off = base + kv_offset(pos, h, head_dim, kv_heads) + d;
      k_cache[off] = k[h * head_dim + d];
      v_cache[off] = v[h * head_dim + d];
    }
}

void kv_fetch_ref(const float* k_cache, const float* v_cache, float* k_out,
                  float* v_out, uint32_t layer, uint32_t pos,
                  uint32_t kv_heads, uint32_t head_dim, uint32_t context) {
  uint32_t layer_stride = context * kv_heads * head_dim;
  uint32_t base = layer * layer_stride;
  for (uint32_t h = 0; h < kv_heads; ++h)
    for (uint32_t d = 0; d < head_dim; ++d) {
      uint32_t off = base + kv_offset(pos, h, head_dim, kv_heads) + d;
      k_out[h * head_dim + d] = k_cache[off];
      v_out[h * head_dim + d] = v_cache[off];
    }
}

void kv_clear_ref(float* cache, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) cache[i] = 0.0f;
}

void gqa_strided_ref(const float* q, const float* k_cache,
                     const float* v_cache, float* out, uint32_t seq_len,
                     uint32_t kv_len, uint32_t num_heads,
                     uint32_t num_kv_heads, uint32_t head_dim,
                     uint32_t kv_stride, float scale) {
  uint32_t rep = num_heads / num_kv_heads;
  for (uint32_t qp = 0; qp < seq_len; ++qp) {
    for (uint32_t h = 0; h < num_heads; ++h) {
      uint32_t kv_h = h / rep;
      size_t q_off = (size_t)qp * num_heads * head_dim + (size_t)h * head_dim;
      size_t kv_off = (size_t)kv_h * head_dim;
      float m = -1.0f / 0.0f; // -inf, as in the Metal source
      for (uint32_t t = 0; t < kv_len; ++t) {
        float s = 0.0f;
        size_t k_off = (size_t)t * kv_stride + kv_off;
        for (uint32_t i = 0; i < head_dim; ++i) s += q[q_off + i] * k_cache[k_off + i];
        s *= scale;
        if (s > m) m = s;
      }
      float l = 0.0f;
      for (uint32_t t = 0; t < kv_len; ++t) {
        float s = 0.0f;
        size_t k_off = (size_t)t * kv_stride + kv_off;
        for (uint32_t i = 0; i < head_dim; ++i) s += q[q_off + i] * k_cache[k_off + i];
        l += std::exp(s * scale - m);
      }
      for (uint32_t i = 0; i < head_dim; ++i) {
        float acc = 0.0f;
        for (uint32_t t = 0; t < kv_len; ++t) {
          size_t k_off = (size_t)t * kv_stride + kv_off;
          size_t v_off = (size_t)t * kv_stride + kv_off;
          float s = 0.0f;
          for (uint32_t j = 0; j < head_dim; ++j)
            s += q[q_off + j] * k_cache[k_off + j];
          float w = std::exp(s * scale - m) / l;
          acc += w * v_cache[v_off + i];
        }
        out[q_off + i] = acc;
      }
    }
  }
}

void gqa_online_ref(const float* q, const float* k_cache,
                    const float* v_cache, float* out_row, uint32_t head,
                    uint32_t q_pos, uint32_t num_heads,
                    uint32_t num_kv_heads, uint32_t head_dim) {
  uint32_t rep = num_heads / num_kv_heads;
  uint32_t kv_head = head / rep;
  uint32_t kv_stride = num_kv_heads * head_dim;
  uint32_t kv_len = q_pos + 1;
  const float* qv = q + (size_t)head * head_dim;
  float m = -1.0f / 0.0f;
  float l = 0.0f;
  float acc[128];
  for (uint32_t i = 0; i < head_dim; ++i) acc[i] = 0.0f;
  float scale = 1.0f / std::sqrt((float)head_dim);
  for (uint32_t p = 0; p < kv_len; ++p) {
    const float* k = k_cache + (size_t)p * kv_stride + (size_t)kv_head * head_dim;
    const float* v = v_cache + (size_t)p * kv_stride + (size_t)kv_head * head_dim;
    float score = 0.0f;
    for (uint32_t i = 0; i < head_dim; ++i) score += qv[i] * k[i];
    score *= scale;
    float new_m = score > m ? score : m;
    float corr = std::exp(m - new_m);
    float pexp = std::exp(score - new_m);
    l = l * corr + pexp;
    for (uint32_t i = 0; i < head_dim; ++i) acc[i] = acc[i] * corr + pexp * v[i];
    m = new_m;
  }
  float inv = 1.0f / l;
  for (uint32_t i = 0; i < head_dim; ++i)
    out_row[(size_t)head * head_dim + i] = acc[i] * inv;
}

void attention_scores_ref(const float* q, const float* k_cache,
                          float* scores, uint32_t heads, uint32_t kv_heads,
                          uint32_t head_dim, uint32_t length) {
  float scale = 1.0f / std::sqrt((float)head_dim);
  for (uint32_t h = 0; h < heads; ++h) {
    uint32_t kv_h = h % kv_heads;
    for (uint32_t pos = 0; pos < length; ++pos) {
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d)
        dot += q[h * head_dim + d] *
               k_cache[kv_offset(pos, kv_h, head_dim, kv_heads) + d];
      scores[h * length + pos] = dot * scale;
    }
  }
}

void stable_softmax_ref(float* scores, uint32_t heads, uint32_t length,
                        uint32_t offset) {
  // Sole causal enforcement in the port: len = offset+1, entries
  // len..length zeroed. attention.metal `causal_mask` is a no-op as
  // written and was deliberately not ported; see the "causal: future
  // positions zeroed" regression test in test_kernel_ref.cpp.
  uint32_t len = offset + 1;
  for (uint32_t h = 0; h < heads; ++h) {
    float* row = scores + (size_t)h * length;
    float maxv = -1.0e30f;
    for (uint32_t i = 0; i < len; ++i) maxv = row[i] > maxv ? row[i] : maxv;
    float sum = 0.0f;
    for (uint32_t i = 0; i < len; ++i) {
      row[i] = std::exp(row[i] - maxv);
      sum += row[i];
    }
    float inv = 1.0f / (sum < 1.0e-8f ? 1.0e-8f : sum);
    for (uint32_t i = 0; i < len; ++i) row[i] *= inv;
    for (uint32_t i = len; i < length; ++i) row[i] = 0.0f;
  }
}

void attention_value_ref(const float* scores, const float* v_cache,
                         float* out, uint32_t heads, uint32_t kv_heads,
                         uint32_t head_dim, uint32_t length,
                         uint32_t offset) {
  uint32_t len = offset + 1;
  for (uint32_t h = 0; h < heads; ++h) {
    uint32_t kv_h = h % kv_heads;
    for (uint32_t d = 0; d < head_dim; ++d) {
      float acc = 0.0f;
      for (uint32_t pos = 0; pos < len; ++pos)
        acc += scores[h * length + pos] *
               v_cache[kv_offset(pos, kv_h, head_dim, kv_heads) + d];
      out[h * head_dim + d] = acc;
    }
  }
}

void swiglu_act_ref(const float* gate, const float* up, uint32_t n,
                    float* act) {
  for (uint32_t i = 0; i < n; ++i) act[i] = silu(gate[i]) * up[i];
}

void swiglu_full_ref(const uint8_t* w_gate, const uint16_t* s_gate,
                     const uint8_t* w_up, const uint16_t* s_up,
                     const float* x, float* out, uint32_t hidden,
                     uint32_t intermediate) {
  std::vector<float> gate(intermediate), up(intermediate);
  int4abi::matvec_fmt(tensorabi::QuantFormat::A_SignedPerRow, w_gate, s_gate,
                      x, gate.data(), hidden, intermediate);
  int4abi::matvec_fmt(tensorabi::QuantFormat::A_SignedPerRow, w_up, s_up,
                      x, up.data(), hidden, intermediate);
  swiglu_act_ref(gate.data(), up.data(), intermediate, out);
}

void residual_ref(const float* a, const float* b, float scale,
                  uint32_t n, float* out) {
  for (uint32_t i = 0; i < n; ++i) out[i] = a[i] + b[i] * scale;
}

bool lora_delta_ref(const float* A, const float* B, const float* X,
                    float* Y, uint32_t rank, uint32_t K, uint32_t N,
                    uint32_t M, float scale) {
  if (rank > 64) return false; // Metal tmp[64] — see header
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float tmp[64];
      for (uint32_t r = 0; r < rank; ++r) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < K; ++k) acc += A[r * K + k] * X[m * K + k];
        tmp[r] = acc;
      }
      float delta = 0.0f;
      for (uint32_t r = 0; r < rank; ++r) delta += B[n * rank + r] * tmp[r];
      Y[m * N + n] += delta * scale;
    }
  }
  return true;
}

void lora_down_ref(const float* x, const uint16_t* A_bits, float* z,
                   uint32_t seq, uint32_t in_dim, uint32_t rank) {
  for (uint32_t ri = 0; ri < rank; ++ri) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < in_dim; ++i)
      acc += x[(size_t)seq * in_dim + i] *
             int4abi::half_to_float(A_bits[(size_t)ri * in_dim + i]);
    z[(size_t)seq * rank + ri] = acc;
  }
}

void lora_up_ref(const float* z, const uint16_t* B_bits, float* y,
                 uint32_t seq, uint32_t out_dim, uint32_t rank,
                 float alpha_over_rank) {
  for (uint32_t o = 0; o < out_dim; ++o) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < rank; ++i)
      acc += z[(size_t)seq * rank + i] *
             int4abi::half_to_float(B_bits[(size_t)o * rank + i]);
    y[(size_t)seq * out_dim + o] += alpha_over_rank * acc;
  }
}

uint32_t argmax_ref(const float* logits, uint32_t vocab) {
  float best = -1.0e30f;
  uint32_t best_idx = 0;
  for (uint32_t i = 0; i < vocab; ++i)
    if (logits[i] > best) { best = logits[i]; best_idx = i; }
  return best_idx;
}

void temperature_softmax_ref(float* logits, uint32_t vocab, float temp) {
  float inv_temp = 1.0f / (temp < 1.0e-6f ? 1.0e-6f : temp);
  float maxv = -1.0e30f;
  for (uint32_t i = 0; i < vocab; ++i) {
    logits[i] *= inv_temp;
    maxv = logits[i] > maxv ? logits[i] : maxv;
  }
  float sum = 0.0f;
  for (uint32_t i = 0; i < vocab; ++i) {
    logits[i] = std::exp(logits[i] - maxv);
    sum += logits[i];
  }
  float inv = 1.0f / (sum < 1.0e-8f ? 1.0e-8f : sum);
  for (uint32_t i = 0; i < vocab; ++i) logits[i] *= inv;
}

void decode_layer_ref(const float* x, const LayerWeightsA& w,
                      const LayerDims& d, uint32_t pos, float* k_cache,
                      float* v_cache, float* out, float* scratch) {
  using namespace int4abi;
  using tensorabi::QuantFormat;
  uint32_t H = d.hidden, I = d.intermediate;
  uint32_t Q = d.q_heads * d.head_dim, KV = d.kv_heads * d.head_dim;
  // scratch layout: norm[H] | q[Q] | k[KV] | v[KV] | attn[Q] | proj[H] |
  //                 resid[H] | norm2[H] | mlp[I]
  float* norm = scratch;
  float* q = norm + H;
  float* k = q + Q;
  float* v = k + KV;
  float* attn = v + KV;
  float* proj = attn + Q;
  float* resid = proj + H;
  float* norm2 = resid + H;
  float* mlp = norm2 + H;

  rmsnorm_ref(x, w.attn_norm_gain, d.eps, H, norm);
  matvec_fmt(QuantFormat::A_SignedPerRow, w.wq, w.sq, norm, q, H, Q);
  matvec_fmt(QuantFormat::A_SignedPerRow, w.wk, w.sk, norm, k, H, KV);
  matvec_fmt(QuantFormat::A_SignedPerRow, w.wv, w.sv, norm, v, H, KV);
  rope_interleaved_ref(q, k, pos, d.q_heads, d.kv_heads, d.head_dim);
  kv_store_ref(k, v, k_cache, v_cache, 0, pos, d.kv_heads, d.head_dim,
               d.context);
  // Attention over the cache as [pos][kv_heads*head_dim] rows:
  // gqa_strided_ref wants kv_stride = kv_heads*head_dim and a
  // per-position q; call it with seq_len 1 and the single q row.
  {
    float scale = 1.0f / std::sqrt((float)d.head_dim);
    gqa_strided_ref(q, k_cache, v_cache, attn, 1, pos + 1, d.q_heads,
                    d.kv_heads, d.head_dim, KV, scale);
  }
  matvec_fmt(QuantFormat::A_SignedPerRow, w.wo, w.so, attn, proj, Q, H);
  residual_ref(x, proj, 1.0f, H, resid);
  rmsnorm_ref(resid, w.ffn_norm_gain, d.eps, H, norm2);
  swiglu_full_ref(w.w_gate, w.s_gate, w.w_up, w.s_up, norm2, mlp, H, I);
  // down projection + residual: out = resid + down(mlp)
  std::vector<float> down(H);
  matvec_fmt(QuantFormat::A_SignedPerRow, w.w_down, w.s_down, mlp,
             down.data(), I, H);
  residual_ref(resid, down.data(), 1.0f, H, out);
}

} // namespace kref
