// vk_dispatch — Phase B dispatch CLI.
//   vk_dispatch list
//   vk_dispatch embedding <embedding.spv>
//   vk_dispatch matvec    <int4_matvec.spv>
// Exit codes: 0 = executed and verified, 1 = executed but mismatch
// or hard error, 3 = BLOCKED (no loader / no device). BLOCKED is a
// report, not a pass: nothing falls back to CPU in this tool.
#include "vulkan_runtime.h"
#include "int4_abi.h"
#include "case_data.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>

static std::vector<uint8_t> read_file(const char* path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in),
                              std::istreambuf_iterator<char>());
}

static const char* type_name(uint32_t t) {
  switch (t) {
    case 1: return "CPU (software)";
    case 2: return "DISCRETE_GPU";
    case 3: return "INTEGRATED_GPU";
    case 4: return "VIRTUAL_GPU";
    default: return "OTHER";
  }
}

static int report_blocked(const vkrun::Status& s) {
  printf("BLOCKED: %s — %s\n", vkrun::code_name(s.code), s.message.c_str());
  return 3;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: vk_dispatch list|embedding <spv>|matvec <spv>\n");
    return 1;
  }
  std::string mode = argv[1];
  if (mode == "list") {
    std::vector<vkrun::DeviceInfo> devs;
    vkrun::Status s = vkrun::list_devices(devs);
    if (!s.ok()) return report_blocked(s);
    if (devs.empty()) {
      printf("BLOCKED: %s — loader present, 0 physical devices (no ICD)\n",
             vkrun::code_name(vkrun::Code::NoPhysicalDevice));
      return 3;
    }
    for (auto& d : devs)
      printf("DEVICE: %s [%s] api %u.%u\n", d.name.c_str(), type_name(d.type),
             (d.api_version >> 22) & 0x3FF, (d.api_version >> 12) & 0x3FF);
    return 0;
  }
  if (argc < 3) { fprintf(stderr, "missing spv path\n"); return 1; }
  auto spv = read_file(argv[2]);
  if (spv.empty()) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }

  vkrun::RunRequest req;
  req.spirv = spv;

  if (mode == "embedding") {
    const uint32_t hidden = 8, vocab = 4, token = 2;
    std::vector<uint32_t> table(vocab * hidden), tokens = {token}, out(hidden);
    for (uint32_t t = 0; t < vocab; ++t)
      for (uint32_t l = 0; l < hidden; ++l)
        table[t * hidden + l] = int4abi::float_to_half(100.0f * t + (float)l + 1.0f);
    auto blob = [](const std::vector<uint32_t>& v) {
      return std::vector<uint8_t>((const uint8_t*)v.data(),
                                  (const uint8_t*)v.data() + v.size() * 4);
    };
    req.buffers = {{blob(table), 0, false}, {blob(tokens), 0, false}, {{}, hidden * 4, true}};
    uint32_t push[1] = {hidden};
    req.push_constants.assign((uint8_t*)push, (uint8_t*)push + 4);
    req.spec_local_size = 256;
    req.groups_x = (hidden + 255) / 256;
    vkrun::RunResult r = vkrun::run_compute(req);
    if (!r.status.ok()) {
      if (r.status.code == vkrun::Code::LoaderMissing ||
          r.status.code == vkrun::Code::NoPhysicalDevice)
        return report_blocked(r.status);
      printf("ERROR: %s — %s (vk %d)\n", vkrun::code_name(r.status.code),
             r.status.message.c_str(), r.status.vk_result);
      return 1;
    }
    printf("DEVICE: %s [%s]\n", r.device.name.c_str(), type_name(r.device.type));
    const uint32_t* got = (const uint32_t*)r.outputs[0].data();
    bool ok = true;
    for (uint32_t l = 0; l < hidden; ++l)
      if (got[l] != table[token * hidden + l]) ok = false;
    printf(ok ? "PASS embedding golden (bitwise, 8 lanes)\n"
              : "FAIL embedding golden mismatch\n");
    return ok ? 0 : 1;
  }

  if (mode == "matvec") {
    auto c = casedata::build_matvec_case();
    auto blob = [](const void* p, size_t n) {
      return std::vector<uint8_t>((const uint8_t*)p, (const uint8_t*)p + n);
    };
    req.buffers = {
        {blob(c.packed.data(), c.packed.size()), 0, false},
        {blob(c.scales.data(), c.scales.size() * 4), 0, false},
        {blob(c.x.data(), c.x.size() * 4), 0, false},
        {{}, c.out_dim * 4, true}};
    uint32_t push[2] = {c.in_dim, c.out_dim};
    req.push_constants.assign((uint8_t*)push, (uint8_t*)push + 8);
    req.spec_local_size = 32;
    req.spec_group = 32;
    req.groups_x = c.out_dim;
    vkrun::RunResult r = vkrun::run_compute(req);
    if (!r.status.ok()) {
      if (r.status.code == vkrun::Code::LoaderMissing ||
          r.status.code == vkrun::Code::NoPhysicalDevice)
        return report_blocked(r.status);
      printf("ERROR: %s — %s (vk %d)\n", vkrun::code_name(r.status.code),
             r.status.message.c_str(), r.status.vk_result);
      return 1;
    }
    printf("DEVICE: %s [%s]\n", r.device.name.c_str(), type_name(r.device.type));
    const float* got = (const float*)r.outputs[0].data();
    bool ok = true;
    for (uint32_t i = 0; i < c.out_dim; ++i) {
      double tol = 1e-4 * (1.0 + std::fabs(c.expected_y[i])); // reduction-order tolerance
      if (std::fabs((double)got[i] - c.expected_y[i]) > tol) ok = false;
    }
    printf(ok ? "PASS matvec golden (within reduction tolerance)\n"
              : "FAIL matvec golden mismatch\n");
    return ok ? 0 : 1;
  }

  fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 1;
}
