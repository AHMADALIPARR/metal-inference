// vulkan_runtime.h — minimal Vulkan compute runtime for the
// Phase B vertical slice. C++17, no Vulkan loader link dependency:
// libvulkan is dlopen'd at runtime so the binary builds anywhere
// and fails explicitly where Vulkan is absent.
//
// Every failure is a typed Status with a named code. A missing
// loader, a missing ICD, or zero physical devices is reported as
// BLOCKED with the precise reason — never as a pass, never as a
// silent CPU fallback.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vkrun {

enum class Code {
  Ok,
  LoaderMissing,   // libvulkan.so.1 could not be dlopen'd
  InstanceFailed,
  NoPhysicalDevice, // loader present, zero devices (no ICD)
  DeviceUnsuitable,
  OutOfMemory,
  IoError,
  VulkanError,
};

struct Status {
  Code code = Code::Ok;
  std::string message;
  int vk_result = 0;
  bool ok() const { return code == Code::Ok; }
};

struct DeviceInfo {
  std::string name;
  uint32_t type = 0; // VkPhysicalDeviceType value
  uint32_t api_version = 0;
};

// Queried compute capabilities of the first compute-capable device.
// Values come from VkPhysicalDeviceProperties, never from an assumed
// vendor or model. query_capabilities returns a non-Ok status
// (LoaderMissing / NoPhysicalDevice / ...) when no device can be
// queried; callers must treat that as "unknown", not as zeros.
struct DeviceCaps {
  uint32_t max_workgroup_count_x = 0;
  uint32_t max_workgroup_count_y = 0;
  uint32_t max_workgroup_count_z = 0;
  uint32_t max_workgroup_invocations = 0;
  uint32_t max_workgroup_size_x = 0;
  uint32_t max_workgroup_size_y = 0;
  uint32_t max_workgroup_size_z = 0;
  uint32_t max_push_constants_size = 0;
  uint32_t max_shared_memory_per_workgroup = 0;
};

Status query_capabilities(DeviceCaps& out);

struct BufferSpec {
  std::vector<uint8_t> initial; // uploaded before dispatch; empty = uninitialized
  size_t size = 0;              // if initial non-empty, size is ignored (= initial.size())
  bool readback = false;
};

struct RunRequest {
  std::vector<uint8_t> spirv;          // shader module bytes
  std::vector<BufferSpec> buffers;     // bindings 0..N-1, all storage buffers
  std::vector<uint8_t> push_constants; // raw push-constant block
  uint32_t spec_local_size = 0;        // spec constant 0 (workgroup size); 0 = leave default
  uint32_t spec_group = 0;             // spec constant 1; 0 = leave default
  uint32_t groups_x = 1, groups_y = 1, groups_z = 1;
};

struct RunResult {
  Status status;
  DeviceInfo device;
  std::vector<std::vector<uint8_t>> outputs; // readback buffers, in binding order
};

// Enumerate devices without dispatching. ok() with an empty list
// means the loader works but no ICD/device exists.
Status list_devices(std::vector<DeviceInfo>& out);

// Execute one compute dispatch end to end: instance, device,
// buffers, descriptor set, pipeline, command buffer, fence,
// readback, cleanup. On any failure, status names the stage.
RunResult run_compute(const RunRequest& req);

const char* code_name(Code c);

} // namespace vkrun
