// command_graph.h — the Phase C transformer command graph.
//
// A Graph is a validated list of kernel launches in forward-pass
// order. Building it checks, per node, that every weight tensor's
// QuantFormat is exactly the format that kernel decodes, that every
// descriptor is valid against its (declared) buffer size, and that
// dispatch dimensions fit the workgroup sizes the shaders use.
// Building NEVER dispatches anything: on a machine with no Vulkan
// device the graph builds, validates, and reports BLOCKED at
// execution time. A constructed graph is not an executed inference
// pass (docs/PHASE_C.md).
//
// Canonical graph instantiation (the one build_forward_graph builds):
// FP16 activations throughout; fused QKV decodes Scheme C; output
// projection, MLP and logits decode Scheme A. Each node's format is
// recorded on the node itself, so a mixed-format graph is visible
// rather than assumed. The Scheme-B stack (int4_transformer.metal)
// is ported as shaders and CPU refs, but its float-activation
// kernels are NOT mixed into this graph: mixing them would
// reinterpret FP16 activation tensors as float. See PHASE_C.md.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "tensor_abi.h"

namespace graph {

struct ModelDims {
  uint32_t hidden = 3072;
  uint32_t intermediate = 8192;
  uint32_t q_heads = 24;
  uint32_t kv_heads = 8;
  uint32_t head_dim = 128;
  uint32_t vocab = 32000;
  uint32_t context = 4096;
  float epsilon = 1.0e-5f;
  // The repo states no default layer count (Swift defaults omit it;
  // types.h only declares the field). Layer count is a graph
  // parameter, never assumed.
};

// One kernel launch. `shader` is the .comp stem; `push` is the raw
// push-constant block the shader documents; `groups_*` are the
// dispatch dimensions; `weight_format` is the format the kernel
// decodes, or None for kernels with no quantized weights.
struct GraphNode {
  std::string shader;
  std::string label;
  tensorabi::QuantFormat weight_format = tensorabi::QuantFormat::None;
  std::vector<tensorabi::TensorDescriptor> inputs;
  std::vector<tensorabi::TensorDescriptor> outputs;
  std::vector<uint8_t> push;
  uint32_t spec_local_size = 256;
  uint32_t spec_group = 0; // spec constant 1 (INT4 group); 0 = unused
  uint32_t groups_x = 1, groups_y = 1, groups_z = 1;
  // Declared buffer sizes, in bytes, one per input/output in order;
  // the builder validates each descriptor against its size.
  std::vector<uint64_t> buffer_bytes;
};

struct Graph {
  std::vector<GraphNode> nodes;
  ModelDims dims;
  uint32_t layers = 0;
  uint32_t seq_len = 0;
  // Capabilities the build could check against. When no Vulkan
  // device exists, caps_known is false and no dispatch dimension is
  // claimed to fit any device: the check is recorded as SKIPPED,
  // not passed.
  bool caps_known = false;
  std::string caps_note;
};

// Build the forward graph for `layers` layers and a decode of
// `seq_len` positions (seq_len 1 = single-token decode). Returns
// false with `why` naming the first failed check: bad dims, a
// format mismatch, an invalid descriptor, or a dispatch that does
// not fit the queried capabilities (when known). On success the
// graph is fully wired: embedding, per-layer
// rmsnorm -> fused_qkv -> rope -> kv_store -> gqa_attention ->
// output_projection -> residual -> rmsnorm -> swiglu -> mlp_down ->
// residual, then final_norm -> logits -> argmax.
bool build_forward_graph(const ModelDims& dims, uint32_t layers,
                         uint32_t seq_len, Graph& out, std::string& why);

// Node count for a graph of `layers` layers: 1 embedding +
// 11 per layer + 3 tail (final_norm, logits, argmax).
inline constexpr uint32_t node_count_for(uint32_t layers) {
  return 4 + 11 * layers;
}

} // namespace graph
