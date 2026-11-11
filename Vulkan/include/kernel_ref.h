// kernel_ref.h — independent CPU references for every Vulkan kernel
// ported in Phase C. Plain C++17, FP32 math, no Python, no GPU.
//
// Each function mirrors one Metal kernel's arithmetic exactly as
// written in Sources/Kernels/ (cited per function). Where Metal
// outputs FP16, the reference returns FP32 and the test applies
// FP16 rounding (float_to_half) before comparing, so half-rounding
// and kernel math are tested separately. Serial accumulation order
// matches the Metal source; a GLSL kernel that reduces in a
// different order is compared with the tolerance stated in
// docs/PHASE_C.md, never claimed bitwise-equal.
//
// The composed decode_layer_ref wires the per-kernel refs in
// forward-pass order for one transformer layer at arbitrary small
// dims; it is the CPU-side image of the Vulkan command graph.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstddef>
#include "int4_abi.h"
#include "tensor_abi.h"

namespace kref {

inline float safe_rsqrt(float x) {
  float v = x < 1.0e-6f ? 1.0e-6f : x;
  return 1.0f / std::sqrt(v);
}
inline float silu(float x) {
  float c = x < -80.0f ? -80.0f : (x > 80.0f ? 80.0f : x);
  return x / (1.0f + std::exp(-c));
}
inline uint32_t kv_offset(uint32_t token, uint32_t kv_head, uint32_t head_dim,
                          uint32_t kv_heads) {
  return ((token * kv_heads) + kv_head) * head_dim;
}

// normalization.metal rmsnorm / final_norm (identical math):
//   inv = safe_rsqrt(mean(x^2) + eps); y[i] = x[i]*inv*gain[i]
void rmsnorm_ref(const float* x, const float* gain, float eps,
                 uint32_t n, float* y);
// int4_transformer.metal rmsNorm (float, eps fixed 1e-6, same math).
inline void rmsnorm_f32_ref(const float* x, const float* gain,
                            uint32_t n, float* y) {
  rmsnorm_ref(x, gain, 1.0e-6f, n, y);
}
// normalization.metal layer_norm (bias may be null -> +0):
//   mean, var = max(E[x^2]-mean^2, 0); y = (x-mean)*rsqrt(var+eps)*gain+bias
void layernorm_ref(const float* x, const float* gain, const float* bias,
                   float eps, uint32_t n, float* y);

// rope.metal rope_qk (interleaved adjacent pairs), in place on the
// flat Q (q_heads*head_dim) and K (kv_heads*head_dim) vectors for
// one token position. K head for Q head h is h % kv_heads.
void rope_interleaved_ref(float* q, float* k, uint32_t pos,
                          uint32_t q_heads, uint32_t kv_heads,
                          uint32_t head_dim);
// gqa_strided.metal rope_apply / int4_transformer.metal applyRoPE
// (half-split pairs): pair d rotates (x[d], x[d+half]) with
// angle = pos * theta^(-2d/head_dim). q/k are [seq][heads][head_dim]
// flat; pos is the sequence position; theta defaults 10000.
void rope_split_ref(float* q, float* k, uint32_t pos, uint32_t seq_len,
                    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                    float theta = 10000.0f);

// kv_cache.metal: store/fetch one position, optionally in layer `layer`
// of a cache with per-layer stride context*kv_heads*head_dim.
void kv_store_ref(const float* k, const float* v, float* k_cache,
                  float* v_cache, uint32_t layer, uint32_t pos,
                  uint32_t kv_heads, uint32_t head_dim, uint32_t context);
void kv_fetch_ref(const float* k_cache, const float* v_cache, float* k_out,
                  float* v_out, uint32_t layer, uint32_t pos,
                  uint32_t kv_heads, uint32_t head_dim, uint32_t context);
// int4_transformer.metal secureClearKV: zero `count` floats.
void kv_clear_ref(float* cache, uint32_t count);

// gqa_strided.metal gqa_attention (FP16 in Metal; FP32 here):
// two-pass (max, then sum of exp), scale applied to scores,
// kv_head = h / (num_heads / num_kv_heads), cache row stride
// kv_stride elements per position. q is [seq][heads][d], cache is
// [pos][kv_stride], out is [seq][heads][d].
void gqa_strided_ref(const float* q, const float* k_cache,
                     const float* v_cache, float* out, uint32_t seq_len,
                     uint32_t kv_len, uint32_t num_heads,
                     uint32_t num_kv_heads, uint32_t head_dim,
                     uint32_t kv_stride, float scale);
// int4_transformer.metal gqaAttention: online softmax with the
// rescaling correction, one (head, qPos) at a time. Cache is
// pos-major with stride num_kv_heads*head_dim. q is the flat qkv
// row for qPos (num_heads*head_dim); out is [qPos][hidden] row.
void gqa_online_ref(const float* q, const float* k_cache,
                    const float* v_cache, float* out_row, uint32_t head,
                    uint32_t q_pos, uint32_t num_heads,
                    uint32_t num_kv_heads, uint32_t head_dim);

// attention.metal decomposed pipeline (single query position,
// len = offset+1 positions are live, the rest are zeroed by softmax):
void attention_scores_ref(const float* q, const float* k_cache,
                          float* scores, uint32_t heads, uint32_t kv_heads,
                          uint32_t head_dim, uint32_t length);
void stable_softmax_ref(float* scores, uint32_t heads, uint32_t length,
                        uint32_t offset);
void attention_value_ref(const float* scores, const float* v_cache,
                         float* out, uint32_t heads, uint32_t kv_heads,
                         uint32_t head_dim, uint32_t length,
                         uint32_t offset);

// mlp.metal swiglu (gate/up are Scheme-A INT4 matvecs) + the
// elementwise int4_transformer.metal swigluFFN form.
void swiglu_act_ref(const float* gate, const float* up, uint32_t n,
                    float* act);
void swiglu_full_ref(const uint8_t* w_gate, const uint16_t* s_gate,
                     const uint8_t* w_up, const uint16_t* s_up,
                     const float* x, float* out, uint32_t hidden,
                     uint32_t intermediate);

// residual.metal: out[i] = a[i] + b[i]*scale (scale 1 = plain add).
void residual_ref(const float* a, const float* b, float scale,
                  uint32_t n, float* out);

// lora_delta.metal apply_lora_delta: Y[m][n] += scale * B[n][r] dot
// (A[r][k] dot X[m][k]). Returns false when rank > 64: the Metal
// kernel's stack tmp[64] cannot hold more (ABI limit, enforced by
// graph validation too).
bool lora_delta_ref(const float* A, const float* B, const float* X,
                    float* Y, uint32_t rank, uint32_t K, uint32_t N,
                    uint32_t M, float scale);
// int4_transformer.metal loraDown / loraUp (A/B are FP16 bit patterns).
void lora_down_ref(const float* x, const uint16_t* A_bits, float* z,
                   uint32_t seq, uint32_t in_dim, uint32_t rank);
void lora_up_ref(const float* z, const uint16_t* B_bits, float* y,
                 uint32_t seq, uint32_t out_dim, uint32_t rank,
                 float alpha_over_rank);

// output.metal token_select: first maximum wins (strict >).
uint32_t argmax_ref(const float* logits, uint32_t vocab);
// output.metal temperature_softmax, in place.
void temperature_softmax_ref(float* logits, uint32_t vocab, float temp);

// ---- composed one-layer decode (decode.metal decode_step, but as
// the graph wires it: separate kernels, Scheme-A weights) ----
// All weights are Scheme A (packed + one FP16 scale per row).
// x: [hidden] input. kv caches: [context][kv_heads*head_dim] each.
// out: [hidden]. Scratch is caller-owned and must hold at least
// hidden + q_heads*head_dim + 2*kv_heads*head_dim + intermediate
// floats; the function uses it only as workspace.
struct LayerWeightsA {
  const uint8_t* wq; const uint16_t* sq;
  const uint8_t* wk; const uint16_t* sk;
  const uint8_t* wv; const uint16_t* sv;
  const uint8_t* wo; const uint16_t* so;
  const uint8_t* w_gate; const uint16_t* s_gate;
  const uint8_t* w_up; const uint16_t* s_up;
  const uint8_t* w_down; const uint16_t* s_down;
  const float* attn_norm_gain;
  const float* ffn_norm_gain;
};
struct LayerDims {
  uint32_t hidden, intermediate, q_heads, kv_heads, head_dim, context;
  float eps = 1.0e-5f;
};
void decode_layer_ref(const float* x, const LayerWeightsA& w,
                      const LayerDims& d, uint32_t pos, float* k_cache,
                      float* v_cache, float* out, float* scratch);

} // namespace kref
