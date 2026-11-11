// check_reflection.cpp — SPIR-V reflection assertions via the
// SPIRV-Reflect C API. Verifies the ABI contract of each compiled
// shader: entry point, descriptor bindings, push-constant size,
// specialization constants. Exit 1 on any failed assertion.
// Usage: check_reflection <spv> <shader-name>
// The table below is the contract: bindings count, push-block bytes,
// whether spec constant 0 (workgroup size) and spec constant 1
// (INT4 group size, Phase B matvec only) must be present.
#include <spirv_reflect.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int g_fail = 0;
static void expect(bool cond, const char* what) {
  printf("%s %s\n", cond ? "PASS" : "FAIL", what);
  if (!cond) ++g_fail;
}

struct Spec {
  const char* name;
  uint32_t bindings;
  uint32_t push_bytes;
  bool spec0, spec1;
};

static const Spec kSpecs[] = {
    {"embedding", 3, 4, true, false},
    {"int4_matvec", 4, 8, true, true},
    {"rmsnorm", 3, 8, true, false},
    {"rmsnorm_f32", 3, 4, true, false},
    {"layernorm", 4, 8, true, false},
    {"rope", 2, 16, true, false},
    {"rope_split", 2, 24, true, false},
    {"fused_qkv", 10, 16, true, false},
    {"int4_matvec_fmt", 4, 12, true, false},
    {"kv_store", 4, 20, true, false},
    {"kv_fetch", 4, 20, true, false},
    {"kv_clear", 1, 4, true, false},
    {"gqa_attention", 4, 28, true, false},
    {"gqa_online", 4, 20, true, false},
    {"attention_scores", 3, 16, true, false},
    {"softmax", 1, 12, true, false},
    {"attention_value", 3, 20, true, false},
    {"swiglu", 6, 12, true, false},
    {"swiglu_act", 3, 4, true, false},
    {"mlp_down", 4, 12, true, false},
    {"residual", 3, 8, true, false},
    {"lora_delta", 4, 16, true, false},
    {"lora_down", 3, 8, true, false},
    {"lora_up", 3, 12, true, false},
    {"logits", 4, 12, true, false},
    {"argmax", 2, 4, true, false},
    {"temperature_softmax", 1, 8, true, false},
};

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: check_reflection <spv> <shader-name>\n");
    return 1;
  }
  const Spec* spec = nullptr;
  for (auto& s : kSpecs)
    if (std::string(s.name) == argv[2]) spec = &s;
  if (!spec) {
    fprintf(stderr, "unknown shader %s\n", argv[2]);
    return 1;
  }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 1;
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
  SpvReflectShaderModule m{};
  if (spvReflectCreateShaderModule(bytes.size(), bytes.data(), &m) !=
      SPV_REFLECT_RESULT_SUCCESS) {
    fprintf(stderr, "spvReflectCreateShaderModule failed\n");
    return 1;
  }
  expect(std::string(m.entry_point_name) == "main", "entry point is 'main'");

  uint32_t nb = 0;
  spvReflectEnumerateDescriptorBindings(&m, &nb, nullptr);
  std::vector<SpvReflectDescriptorBinding*> binds(nb);
  spvReflectEnumerateDescriptorBindings(&m, &nb, binds.data());
  {
    char buf[128];
    snprintf(buf, sizeof buf, "%u descriptor bindings", spec->bindings);
    expect(nb == spec->bindings, buf);
  }
  bool layout_ok = nb == spec->bindings;
  for (uint32_t i = 0; i < nb && layout_ok; ++i) {
    if (binds[i]->set != 0 || binds[i]->binding != i) layout_ok = false;
    if (binds[i]->descriptor_type != SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      layout_ok = false;
  }
  expect(layout_ok, "bindings are set 0, sequential from 0, all storage buffers");

  uint32_t np = 0;
  spvReflectEnumeratePushConstantBlocks(&m, &np, nullptr);
  expect(np == 1, "exactly one push-constant block");
  if (np == 1) {
    SpvReflectBlockVariable* blk = nullptr;
    spvReflectEnumeratePushConstantBlocks(&m, &np, &blk);
    char buf[128];
    snprintf(buf, sizeof buf, "push block is %u bytes", spec->push_bytes);
    expect(blk && blk->size == spec->push_bytes, buf);
  }

  uint32_t ns = 0;
  SpvReflectResult sr = spvReflectEnumerateSpecializationConstants(&m, &ns, nullptr);
  if (sr == SPV_REFLECT_RESULT_SUCCESS) {
    std::vector<SpvReflectSpecializationConstant*> specs(ns);
    spvReflectEnumerateSpecializationConstants(&m, &ns, specs.data());
    bool has0 = false, has1 = false;
    for (auto* s : specs) {
      if (s->constant_id == 0) has0 = true;
      if (s->constant_id == 1) has1 = true;
    }
    if (spec->spec0) expect(has0, "spec constant 0 (workgroup size) present");
    if (spec->spec1) expect(has1, "spec constant 1 (group size) present");
  } else {
    expect(false, "spec constant enumeration");
  }

  spvReflectDestroyShaderModule(&m);
  return g_fail ? 1 : 0;
}
