// tensor_abi.cpp — descriptor validation + format compatibility.
// See tensor_abi.h for the format definitions.
#include "tensor_abi.h"

namespace tensorabi {

bool validate_descriptor(const TensorDescriptor& d, uint64_t buffer_bytes,
                         std::string& why) {
  if (d.rank < 1 || d.rank > 4) { why = "rank must be 1..4"; return false; }
  for (uint32_t i = 0; i < d.rank; ++i)
    if (d.shape[i] == 0) { why = "shape dim 0 is empty"; return false; }
  // Row-major contiguous strides. For a quantized weight the
  // storage stride is in packed bytes: row stride =
  // ceil(in_dim/2), lane stride 1 byte.
  {
    uint32_t expect[4] = {0, 0, 0, 0};
    if (d.format != QuantFormat::None && d.rank == 2) {
      expect[0] = (uint32_t)packed_row_bytes(d.shape[1]);
      expect[1] = 1;
    } else {
      contiguous_strides(d.rank, d.shape, expect);
    }
    for (uint32_t i = 0; i < d.rank; ++i)
      if (d.strides[i] != expect[i]) {
        why = "strides are not row-major contiguous";
        return false;
      }
  }
  if (d.alignment == 0 || (d.alignment & (d.alignment - 1)) != 0) {
    why = "alignment must be a nonzero power of two";
    return false;
  }
  if (d.buffer_offset % d.alignment != 0) {
    why = "buffer_offset is not aligned";
    return false;
  }
  uint64_t need = d.data_bytes() + d.scale_bytes();
  if (d.buffer_offset > buffer_bytes || need > buffer_bytes - d.buffer_offset) {
    why = "descriptor extends past the end of its buffer";
    return false;
  }
  if (d.group_size != format_group_size(d.format)) {
    why = std::string("group_size does not match format ") + format_name(d.format);
    return false;
  }
  if (d.scale_layout != format_scale_layout(d.format)) {
    why = std::string("scale_layout does not match format ") + format_name(d.format);
    return false;
  }
  if (d.format != QuantFormat::None && d.scale_dtype != DType::F16) {
    why = "quantized scales must be FP16";
    return false;
  }
  if (d.format != QuantFormat::None) {
    if (d.dtype != DType::U8) {
      why = "quantized weight data must be U8 (packed nibbles)";
      return false;
    }
    if (d.rank != 2) {
      why = "quantized weights must be 2-D [rows, in_dim]";
      return false;
    }
    uint32_t in_dim = d.shape[1];
    if ((in_dim & 1u) != 0) {
      why = "quantized in_dim must be even (2 nibbles per byte)";
      return false;
    }
    if (d.format == QuantFormat::B_Offset75PerGroup32 && (in_dim % 32u) != 0) {
      why = "format B requires in_dim % 32 == 0 (int4_transformer asserts this)";
      return false;
    }
    if (d.format == QuantFormat::C_SignedPerGroup64 && (in_dim % 64u) != 0) {
      why = "format C requires in_dim % 64 == 0 (fused_qkv_tiled group math)";
      return false;
    }
  }
  return true;
}

bool check_format_compatible(const TensorDescriptor& weight,
                             QuantFormat expected, std::string& why) {
  if (weight.format != expected) {
    why = std::string("weight format is ") + format_name(weight.format) +
          " but the kernel decodes " + format_name(expected) +
          "; refusing to reinterpret weights";
    return false;
  }
  return true;
}

} // namespace tensorabi
