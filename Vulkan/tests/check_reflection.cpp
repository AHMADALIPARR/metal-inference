// check_reflection.cpp — SPIR-V reflection assertions via the
// SPIRV-Reflect C API. Verifies the ABI contract of each compiled
// shader: entry point, descriptor bindings, push-constant size,
// specialization constants. Exit 1 on any failed assertion.
// Usage: check_reflection <spv> embedding|matvec
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

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: check_reflection <spv> embedding|matvec\n"); return 1; }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
  SpvReflectShaderModule m{};
  if (spvReflectCreateShaderModule(bytes.size(), bytes.data(), &m) != SPV_REFLECT_RESULT_SUCCESS) {
    fprintf(stderr, "spvReflectCreateShaderModule failed\n");
    return 1;
  }
  std::string mode = argv[2];
  expect(std::string(m.entry_point_name) == "main", "entry point is 'main'");

  uint32_t nb = 0;
  spvReflectEnumerateDescriptorBindings(&m, &nb, nullptr);
  std::vector<SpvReflectDescriptorBinding*> binds(nb);
  spvReflectEnumerateDescriptorBindings(&m, &nb, binds.data());
  uint32_t want = mode == "embedding" ? 3 : 4;
  expect(nb == want, mode == "embedding" ? "3 descriptor bindings" : "4 descriptor bindings");
  bool layout_ok = nb == want;
  for (uint32_t i = 0; i < nb && layout_ok; ++i) {
    if (binds[i]->set != 0 || binds[i]->binding != i) layout_ok = false;
    if (binds[i]->descriptor_type != SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER) layout_ok = false;
  }
  expect(layout_ok, "bindings are set 0, sequential from 0, all storage buffers");

  uint32_t np = 0;
  spvReflectEnumeratePushConstantBlocks(&m, &np, nullptr);
  expect(np == 1, "exactly one push-constant block");
  if (np == 1) {
    SpvReflectBlockVariable* blk = nullptr;
    spvReflectEnumeratePushConstantBlocks(&m, &np, &blk);
    uint32_t want_size = mode == "embedding" ? 4 : 8;
    expect(blk && blk->size == want_size,
           mode == "embedding" ? "push block is 4 bytes (hidden)" : "push block is 8 bytes (in_dim, out_dim)");
  }

  uint32_t ns = 0;
  SpvReflectResult sr = spvReflectEnumerateSpecializationConstants(&m, &ns, nullptr);
  if (sr == SPV_REFLECT_RESULT_SUCCESS) {
    std::vector<SpvReflectSpecializationConstant*> specs(ns);
    spvReflectEnumerateSpecializationConstants(&m, &ns, specs.data());
    bool has0 = false, has1 = false;
    for (auto* s : specs) { if (s->constant_id == 0) has0 = true; if (s->constant_id == 1) has1 = true; }
    expect(has0, "spec constant 0 (workgroup size) present");
    if (mode == "matvec") expect(has1, "spec constant 1 (group size) present");
  } else {
    expect(false, "spec constant enumeration");
  }

  spvReflectDestroyShaderModule(&m);
  return g_fail ? 1 : 0;
}
