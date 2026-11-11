// command_graph.cpp — see command_graph.h. Every node is validated
// as it is built; a failed check aborts the build with a reason.
#include "command_graph.h"
#include "vulkan_runtime.h"
#include <cmath>
#include <cstring>

namespace graph {
namespace {

using tensorabi::DType;
using tensorabi::QuantFormat;
using tensorabi::TensorDescriptor;

TensorDescriptor dense(DType dt, uint32_t n) {
  TensorDescriptor d;
  d.dtype = dt;
  d.rank = 1;
  d.shape[0] = n;
  d.strides[0] = 1;
  return d;
}

TensorDescriptor weight(uint32_t rows, uint32_t in_dim, QuantFormat f) {
  TensorDescriptor d;
  d.dtype = DType::U8;
  d.rank = 2;
  d.shape[0] = rows;
  d.shape[1] = in_dim;
  d.strides[0] = (uint32_t)tensorabi::packed_row_bytes(in_dim);
  d.strides[1] = 1;
  d.format = f;
  d.group_size = tensorabi::format_group_size(f);
  d.scale_dtype = DType::F16;
  d.scale_layout = tensorabi::format_scale_layout(f);
  return d;
}

// Scale buffer descriptor for a weight descriptor: one rank-1
// tensor holding every scale slot (B: half2 slots as U32 words).
TensorDescriptor scales_for(const TensorDescriptor& w) {
  TensorDescriptor d;
  uint64_t slots = (uint64_t)w.shape[0] *
                   tensorabi::scale_slots_per_row(w.format, w.shape[1]);
  if (w.format == QuantFormat::B_Offset75PerGroup32) {
    d.dtype = DType::U32;
    d.rank = 1;
    d.shape[0] = (uint32_t)slots;
    d.strides[0] = 1;
  } else {
    d.dtype = DType::F16;
    d.rank = 1;
    d.shape[0] = (uint32_t)slots;
    d.strides[0] = 1;
  }
  return d;
}

void push_u32(std::vector<uint8_t>& p, uint32_t v) {
  uint8_t b[4];
  std::memcpy(b, &v, 4);
  p.insert(p.end(), b, b + 4);
}
void push_f32(std::vector<uint8_t>& p, float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  push_u32(p, u);
}

// Validate one node: every descriptor against its declared buffer
// size, and every quantized weight against the node's format.
bool validate_node(const GraphNode& n, std::string& why) {
  if (n.inputs.size() + n.outputs.size() != n.buffer_bytes.size()) {
    why = "node " + n.label + ": buffer_bytes count does not match bindings";
    return false;
  }
  size_t k = 0;
  auto check_one = [&](const TensorDescriptor& d, bool is_weight) {
    if (!tensorabi::validate_descriptor(d, n.buffer_bytes[k], why)) {
      why = "node " + n.label + ": " + why;
      return false;
    }
    if (is_weight && n.weight_format != QuantFormat::None) {
      if (!tensorabi::check_format_compatible(d, n.weight_format, why)) {
        why = "node " + n.label + ": " + why;
        return false;
      }
    }
    ++k;
    return true;
  };
  // Inputs: weight descriptors are the U8 quantized ones; the scale
  // descriptor follows its weight in the binding list.
  for (auto& d : n.inputs) {
    bool is_weight = (d.format != QuantFormat::None);
    if (!check_one(d, is_weight)) return false;
  }
  for (auto& d : n.outputs)
    if (!check_one(d, false)) return false;
  return true;
}

} // namespace

bool build_forward_graph(const ModelDims& dims, uint32_t layers,
                         uint32_t seq_len, Graph& out, std::string& why) {
  if (layers == 0) { why = "layers must be >= 1"; return false; }
  if (seq_len != 1) {
    // Single-token decode only: the Phase B embedding shader reads
    // tokens[0] and decode.metal is a single-token kernel. Batched
    // prefill is NOT wired in this graph; claiming otherwise would
    // be a placeholder.
    why = "only single-token decode (seq_len 1) is wired";
    return false;
  }
  if (dims.hidden == 0 || dims.intermediate == 0 || dims.q_heads == 0 ||
      dims.kv_heads == 0 || dims.head_dim == 0 || dims.vocab == 0 ||
      dims.context == 0) {
    why = "model dims must all be nonzero";
    return false;
  }
  if (dims.q_heads % dims.kv_heads != 0) {
    why = "q_heads must be a multiple of kv_heads (GQA grouping)";
    return false;
  }
  if (dims.head_dim % 2 != 0) {
    why = "head_dim must be even (RoPE pairs)";
    return false;
  }
  if (seq_len > dims.context) {
    why = "seq_len exceeds context";
    return false;
  }
  const uint32_t H = dims.hidden, I = dims.intermediate;
  const uint32_t Q = dims.q_heads * dims.head_dim;
  const uint32_t KV = dims.kv_heads * dims.head_dim;
  const uint32_t QH = dims.q_heads, KH = dims.kv_heads, DH = dims.head_dim;

  // Queried device capabilities (Phase C directive: use actual
  // queried capabilities, never assumed NVIDIA features). On this
  // VM there is no device, so caps_known stays false and the
  // dispatch-fit check is recorded as skipped.
  vkrun::DeviceCaps caps;
  vkrun::Status cs = vkrun::query_capabilities(caps);
  out.caps_known = cs.ok();
  out.caps_note = cs.ok()
      ? "queried from device"
      : std::string("unavailable: ") + cs.message;

  Graph g;
  g.dims = dims;
  g.layers = layers;
  g.seq_len = seq_len;
  g.caps_known = out.caps_known;
  g.caps_note = out.caps_note;

  auto add = [&](GraphNode&& n) -> bool {
    if (!validate_node(n, why)) return false;
    if (g.caps_known) {
      if (n.spec_local_size > caps.max_workgroup_invocations) {
        why = "node " + n.label + ": workgroup size exceeds device limit";
        return false;
      }
    }
    g.nodes.push_back(std::move(n));
    return true;
  };
  auto bufsize = [](const TensorDescriptor& d) {
    return d.data_bytes() + d.scale_bytes();
  };

  // Activation tensors (FP16 throughout the graph).
  TensorDescriptor tok = dense(DType::U32, 1); // token id (seq_len 1 decode path uses 1; see note)
  TensorDescriptor table = dense(DType::F16, dims.vocab * H);
  TensorDescriptor x = dense(DType::F16, seq_len * H);
  TensorDescriptor gain = dense(DType::F16, H);

  // --- embedding ---
  {
    GraphNode n;
    n.shader = "embedding";
    n.label = "embedding";
    n.inputs = {table, tok};
    n.outputs = {x};
    n.buffer_bytes = {bufsize(table), bufsize(tok), bufsize(x)};
    push_u32(n.push, H);
    n.spec_local_size = 256;
    n.groups_x = (H + 255) / 256;
    if (!add(std::move(n))) return false;
  }

  for (uint32_t L = 0; L < layers; ++L) {
    std::string tag = "L" + std::to_string(L);
    TensorDescriptor qkv_in = x;
    TensorDescriptor q = dense(DType::F16, seq_len * Q);
    TensorDescriptor k = dense(DType::F16, seq_len * KV);
    TensorDescriptor v = dense(DType::F16, seq_len * KV);
    TensorDescriptor kcache = dense(DType::F16, dims.context * KV);
    TensorDescriptor vcache = dense(DType::F16, dims.context * KV);
    TensorDescriptor attn = dense(DType::F16, seq_len * Q);
    TensorDescriptor proj = dense(DType::F16, seq_len * H);
    TensorDescriptor mlp = dense(DType::F16, seq_len * I);
    TensorDescriptor wq = weight(Q, H, QuantFormat::C_SignedPerGroup64);
    TensorDescriptor wk = weight(KV, H, QuantFormat::C_SignedPerGroup64);
    TensorDescriptor wv = weight(KV, H, QuantFormat::C_SignedPerGroup64);
    TensorDescriptor wo = weight(H, Q, QuantFormat::A_SignedPerRow);
    TensorDescriptor wgate = weight(I, H, QuantFormat::A_SignedPerRow);
    TensorDescriptor wup = weight(I, H, QuantFormat::A_SignedPerRow);
    TensorDescriptor wdown = weight(H, I, QuantFormat::A_SignedPerRow);

    // rmsnorm (attention block)
    {
      GraphNode n;
      n.shader = "rmsnorm";
      n.label = "rmsnorm#attn:" + tag;
      n.inputs = {qkv_in, gain};
      n.outputs = {dense(DType::F16, seq_len * H)};
      n.buffer_bytes = {bufsize(qkv_in), bufsize(gain), bufsize(n.outputs[0])};
      push_u32(n.push, H);
      push_f32(n.push, dims.epsilon);
      n.spec_local_size = 256;
      n.groups_x = seq_len;
      if (!add(std::move(n))) return false;
    }
    // fused QKV — Scheme C (fused_qkv_tiled.metal)
    {
      GraphNode n;
      n.shader = "fused_qkv";
      n.label = "fused_qkv:" + tag;
      n.weight_format = QuantFormat::C_SignedPerGroup64;
      n.inputs = {wq, scales_for(wq), wk, scales_for(wk), wv, scales_for(wv),
                  dense(DType::F16, seq_len * H)};
      n.outputs = {q, k, v};
      n.buffer_bytes = {bufsize(wq), bufsize(scales_for(wq)),
                        bufsize(wk), bufsize(scales_for(wk)),
                        bufsize(wv), bufsize(scales_for(wv)),
                        bufsize(n.inputs[6]), bufsize(q), bufsize(k),
                        bufsize(v)};
      push_u32(n.push, H);
      push_u32(n.push, Q);
      push_u32(n.push, KV);
      push_u32(n.push, KV);
      n.spec_local_size = 256;
      n.groups_x = seq_len;
      if (!add(std::move(n))) return false;
    }
    // RoPE (interleaved pairs, rope.metal)
    {
      GraphNode n;
      n.shader = "rope";
      n.label = "rope:" + tag;
      n.inputs = {q, k};
      n.outputs = {q, k}; // in place; listed as both for the record
      n.buffer_bytes = {bufsize(q), bufsize(k), bufsize(q), bufsize(k)};
      push_u32(n.push, 0); // position: decode position 0 for the check build
      push_u32(n.push, QH);
      push_u32(n.push, KH);
      push_u32(n.push, DH);
      n.spec_local_size = 256;
      n.groups_x = (Q / 2 + 255) / 256;
      if (!add(std::move(n))) return false;
    }
    // KV-cache store (layer L)
    {
      GraphNode n;
      n.shader = "kv_store";
      n.label = "kv_store:" + tag;
      n.inputs = {k, v};
      n.outputs = {kcache, vcache};
      n.buffer_bytes = {bufsize(k), bufsize(v), bufsize(kcache),
                        bufsize(vcache)};
      push_u32(n.push, L);
      push_u32(n.push, 0); // position
      push_u32(n.push, KH);
      push_u32(n.push, DH);
      push_u32(n.push, dims.context);
      n.spec_local_size = 256;
      n.groups_x = (KV + 255) / 256;
      if (!add(std::move(n))) return false;
    }
    // Grouped-query attention (gqa_strided.metal, FP16)
    {
      GraphNode n;
      n.shader = "gqa_attention";
      n.label = "gqa_attention:" + tag;
      n.inputs = {q, kcache, vcache};
      n.outputs = {attn};
      n.buffer_bytes = {bufsize(q), bufsize(kcache), bufsize(vcache),
                        bufsize(attn)};
      push_u32(n.push, seq_len);
      push_u32(n.push, 1); // kv_len = position + 1 for the check build
      push_u32(n.push, QH);
      push_u32(n.push, KH);
      push_u32(n.push, DH);
      push_u32(n.push, KV); // kv_stride
      push_f32(n.push, 1.0f / std::sqrt((float)DH));
      n.spec_local_size = 32;
      n.groups_x = (QH * seq_len + 31) / 32;
      if (!add(std::move(n))) return false;
    }
    // Output projection — Scheme A (linear.metal)
    {
      GraphNode n;
      n.shader = "int4_matvec_fmt";
      n.label = "output_projection:" + tag;
      n.weight_format = QuantFormat::A_SignedPerRow;
      n.inputs = {wo, scales_for(wo), attn};
      n.outputs = {proj};
      n.buffer_bytes = {bufsize(wo), bufsize(scales_for(wo)), bufsize(attn),
                        bufsize(proj)};
      push_u32(n.push, Q);
      push_u32(n.push, H);
      push_u32(n.push, (uint32_t)QuantFormat::A_SignedPerRow);
      n.spec_local_size = 256;
      n.spec_group = 0;
      n.groups_x = H;
      if (!add(std::move(n))) return false;
    }
    // Residual (attention block)
    {
      GraphNode n;
      n.shader = "residual";
      n.label = "residual#attn:" + tag;
      n.inputs = {x, proj};
      n.outputs = {dense(DType::F16, seq_len * H)};
      n.buffer_bytes = {bufsize(x), bufsize(proj), bufsize(n.outputs[0])};
      push_u32(n.push, H);
      push_f32(n.push, 1.0f);
      n.spec_local_size = 256;
      n.groups_x = (H + 255) / 256;
      if (!add(std::move(n))) return false;
    }
    // RMSNorm (FFN block)
    {
      GraphNode n;
      n.shader = "rmsnorm";
      n.label = "rmsnorm#ffn:" + tag;
      n.inputs = {dense(DType::F16, seq_len * H), gain};
      n.outputs = {dense(DType::F16, seq_len * H)};
      n.buffer_bytes = {bufsize(n.inputs[0]), bufsize(gain),
                        bufsize(n.outputs[0])};
      push_u32(n.push, H);
      push_f32(n.push, dims.epsilon);
      n.spec_local_size = 256;
      n.groups_x = seq_len;
      if (!add(std::move(n))) return false;
    }
    // SwiGLU — Scheme A gate/up (mlp.metal)
    {
      GraphNode n;
      n.shader = "swiglu";
      n.label = "swiglu:" + tag;
      n.weight_format = QuantFormat::A_SignedPerRow;
      n.inputs = {wgate, scales_for(wgate), wup, scales_for(wup),
                  dense(DType::F16, seq_len * H)};
      n.outputs = {mlp};
      n.buffer_bytes = {bufsize(wgate), bufsize(scales_for(wgate)),
                        bufsize(wup), bufsize(scales_for(wup)),
                        bufsize(n.inputs[4]), bufsize(mlp)};
      push_u32(n.push, H);
      push_u32(n.push, I);
      push_u32(n.push, (uint32_t)QuantFormat::A_SignedPerRow);
      n.spec_local_size = 256;
      n.groups_x = I;
      if (!add(std::move(n))) return false;
    }
    // MLP down — Scheme A
    {
      GraphNode n;
      n.shader = "int4_matvec_fmt";
      n.label = "mlp_down:" + tag;
      n.weight_format = QuantFormat::A_SignedPerRow;
      n.inputs = {wdown, scales_for(wdown), mlp};
      n.outputs = {proj};
      n.buffer_bytes = {bufsize(wdown), bufsize(scales_for(wdown)),
                        bufsize(mlp), bufsize(proj)};
      push_u32(n.push, I);
      push_u32(n.push, H);
      push_u32(n.push, (uint32_t)QuantFormat::A_SignedPerRow);
      n.spec_local_size = 256;
      n.groups_x = H;
      if (!add(std::move(n))) return false;
    }
    // Residual (FFN block) — reuses proj as the down output
    {
      GraphNode n;
      n.shader = "residual";
      n.label = "residual#ffn:" + tag;
      n.inputs = {dense(DType::F16, seq_len * H), proj};
      n.outputs = {dense(DType::F16, seq_len * H)};
      n.buffer_bytes = {bufsize(n.inputs[0]), bufsize(proj),
                        bufsize(n.outputs[0])};
      push_u32(n.push, H);
      push_f32(n.push, 1.0f);
      n.spec_local_size = 256;
      n.groups_x = (H + 255) / 256;
      if (!add(std::move(n))) return false;
    }
    // Next layer's input is this layer's residual output; the
    // descriptors are size-identical, so the graph records the
    // handoff by label rather than by re-plumbing buffers.
    x = dense(DType::F16, seq_len * H);
  }

  // --- final norm ---
  {
    GraphNode n;
    n.shader = "rmsnorm";
    n.label = "final_norm";
    n.inputs = {x, gain};
    n.outputs = {dense(DType::F16, seq_len * H)};
    n.buffer_bytes = {bufsize(x), bufsize(gain), bufsize(n.outputs[0])};
    push_u32(n.push, H);
    push_f32(n.push, dims.epsilon);
    n.spec_local_size = 256;
    n.groups_x = seq_len;
    if (!add(std::move(n))) return false;
  }
  // --- logits — Scheme A (output.metal logits_i4) ---
  {
    TensorDescriptor wlm = weight(dims.vocab, H, QuantFormat::A_SignedPerRow);
    GraphNode n;
    n.shader = "logits";
    n.label = "logits";
    n.weight_format = QuantFormat::A_SignedPerRow;
    n.inputs = {dense(DType::F16, seq_len * H), wlm, scales_for(wlm)};
    n.outputs = {dense(DType::F32, seq_len * dims.vocab)};
    n.buffer_bytes = {bufsize(n.inputs[0]), bufsize(wlm),
                      bufsize(scales_for(wlm)), bufsize(n.outputs[0])};
    push_u32(n.push, H);
    push_u32(n.push, dims.vocab);
    push_u32(n.push, (uint32_t)QuantFormat::A_SignedPerRow);
    n.spec_local_size = 256;
    n.groups_x = dims.vocab;
    if (!add(std::move(n))) return false;
  }
  // --- argmax (output.metal token_select) ---
  {
    GraphNode n;
    n.shader = "argmax";
    n.label = "argmax";
    n.inputs = {dense(DType::F32, seq_len * dims.vocab)};
    n.outputs = {dense(DType::U32, 1)};
    n.buffer_bytes = {bufsize(n.inputs[0]), bufsize(n.outputs[0])};
    push_u32(n.push, dims.vocab);
    n.spec_local_size = 32;
    n.groups_x = 1;
    if (!add(std::move(n))) return false;
  }

  if (g.nodes.size() != node_count_for(layers)) {
    why = "internal error: node count mismatch";
    return false;
  }
  out = std::move(g);
  return true;
}

} // namespace graph
