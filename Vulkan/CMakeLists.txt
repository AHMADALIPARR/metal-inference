# Vulkan/ — Phase B subsystem (INT4 oracle, Vulkan compute runtime,
# GLSL->SPIR-V shaders, verification) extended in Phase C with the
# three-format INT4 ABI, per-kernel CPU references, the full shader
# set, and the transformer command graph. Independent of the Swift
# package (which neither phase modifies).
set(VULKAN_ENV "$ENV{HOME}/workspace/.tools/vulkan-env" CACHE PATH "shader toolchain env")
if(DEFINED ENV{VULKAN_ENV})
  set(VULKAN_ENV "$ENV{VULKAN_ENV}")
endif()
find_program(GLSLANG glslangValidator HINTS ${VULKAN_ENV}/bin REQUIRED)
find_program(SPIRV_VAL spirv-val HINTS ${VULKAN_ENV}/bin REQUIRED)
find_path(VK_INCLUDE vulkan/vulkan.h HINTS ${VULKAN_ENV}/include REQUIRED)
find_path(REFLECT_INCLUDE spirv_reflect.h HINTS ${VULKAN_ENV}/include REQUIRED)
find_library(REFLECT_LIB spirv-reflect HINTS ${VULKAN_ENV}/lib REQUIRED)

add_library(int4_oracle src/int4_oracle.cpp)
target_include_directories(int4_oracle PUBLIC include)

add_library(tensor_abi src/tensor_abi.cpp)
target_include_directories(tensor_abi PUBLIC include)

add_library(kernel_ref src/kernel_ref.cpp)
target_include_directories(kernel_ref PUBLIC include)
target_link_libraries(kernel_ref PUBLIC int4_oracle tensor_abi)

add_library(vulkan_runtime src/vulkan_runtime.cpp)
target_include_directories(vulkan_runtime PUBLIC include PRIVATE ${VK_INCLUDE})
target_link_libraries(vulkan_runtime PUBLIC ${CMAKE_DL_LIBS})

add_library(command_graph src/command_graph.cpp)
target_include_directories(command_graph PUBLIC include)
target_link_libraries(command_graph PUBLIC tensor_abi vulkan_runtime)

add_executable(vk_dispatch tools/vk_dispatch.cpp)
target_link_libraries(vk_dispatch PRIVATE vulkan_runtime int4_oracle
                      tensor_abi command_graph)
target_include_directories(vk_dispatch PRIVATE tests)

add_executable(gen_golden tests/gen_golden.cpp)
target_link_libraries(gen_golden PRIVATE int4_oracle)
target_include_directories(gen_golden PRIVATE tests)

add_executable(gen_kernel_golden tests/gen_kernel_golden.cpp)
target_link_libraries(gen_kernel_golden PRIVATE int4_oracle kernel_ref)
target_include_directories(gen_kernel_golden PRIVATE tests)

add_executable(test_int4_oracle tests/test_int4_oracle.cpp)
target_link_libraries(test_int4_oracle PRIVATE int4_oracle)
target_include_directories(test_int4_oracle PRIVATE tests)

add_executable(test_kernel_ref tests/test_kernel_ref.cpp)
target_link_libraries(test_kernel_ref PRIVATE int4_oracle tensor_abi
                      kernel_ref command_graph)
target_include_directories(test_kernel_ref PRIVATE tests)

# The conda spirv_reflect.h includes "./include/spirv/unified1/spirv.h"
# relative to a repo-root layout its packaging does not preserve.
# Shim it: expose the SPIRV-Headers tree under the expected path.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/reflect_shim/include/spirv/unified1)
file(COPY ${VULKAN_ENV}/include/spirv/unified1/spirv.h
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/reflect_shim/include/spirv/unified1)
add_executable(check_reflection tests/check_reflection.cpp)
target_include_directories(check_reflection PRIVATE ${REFLECT_INCLUDE}
                           ${CMAKE_CURRENT_BINARY_DIR}/reflect_shim)
target_link_libraries(check_reflection PRIVATE ${REFLECT_LIB})
set_target_properties(check_reflection PROPERTIES BUILD_RPATH ${VULKAN_ENV}/lib)

# --- shaders: GLSL -> SPIR-V (generation rule; binaries also checked in) ---
# Phase B: embedding, int4_matvec. Phase C: every remaining kernel.
set(ALL_SHADERS
  embedding int4_matvec
  rmsnorm rmsnorm_f32 layernorm
  rope rope_split
  fused_qkv int4_matvec_fmt
  kv_store kv_fetch kv_clear
  gqa_attention gqa_online
  attention_scores softmax attention_value
  swiglu swiglu_act mlp_down
  residual
  lora_delta lora_down lora_up
  logits argmax temperature_softmax)
foreach(shader ${ALL_SHADERS})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${shader}.spv
    COMMAND ${GLSLANG} -V --target-env vulkan1.2
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp
            -o ${CMAKE_CURRENT_BINARY_DIR}/${shader}.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp
    COMMENT "glslang ${shader}.comp -> ${shader}.spv")
  add_custom_target(spv_${shader} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${shader}.spv)
endforeach()

# --- golden vectors (checked-in copies are the ones tests consume) ---
set(GOLDEN ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/int4_golden.txt)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/int4_golden.regen.txt
  COMMAND gen_golden ${CMAKE_CURRENT_BINARY_DIR}/int4_golden.regen.txt
  DEPENDS gen_golden)
add_custom_target(golden_regen ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/int4_golden.regen.txt)

set(KGOLDEN ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/kernel_golden.txt)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel_golden.regen.txt
  COMMAND gen_kernel_golden ${CMAKE_CURRENT_BINARY_DIR}/kernel_golden.regen.txt
  DEPENDS gen_kernel_golden)
add_custom_target(kernel_golden_regen ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernel_golden.regen.txt)

# --- graph manifests (checked-in copies are the consumer contract) ---
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/graph_manifest_1layer.regen.json
  COMMAND vk_dispatch manifest 1 ${CMAKE_CURRENT_BINARY_DIR}/graph_manifest_1layer.regen.json ${CMAKE_CURRENT_SOURCE_DIR}/shaders
  DEPENDS vk_dispatch)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/graph_manifest_2layers.regen.json
  COMMAND vk_dispatch manifest 2 ${CMAKE_CURRENT_BINARY_DIR}/graph_manifest_2layers.regen.json ${CMAKE_CURRENT_SOURCE_DIR}/shaders
  DEPENDS vk_dispatch)
add_custom_target(manifest_regen ALL DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/graph_manifest_1layer.regen.json
  ${CMAKE_CURRENT_BINARY_DIR}/graph_manifest_2layers.regen.json)

# --- tests ---
add_test(NAME int4_oracle COMMAND test_int4_oracle ${GOLDEN})
add_test(NAME kernel_ref COMMAND test_kernel_ref ${KGOLDEN})
add_test(NAME golden_reproducible
  COMMAND ${CMAKE_COMMAND} -E compare_files ${GOLDEN} ${CMAKE_CURRENT_BINARY_DIR}/int4_golden.regen.txt)
add_test(NAME kernel_golden_reproducible
  COMMAND ${CMAKE_COMMAND} -E compare_files ${KGOLDEN} ${CMAKE_CURRENT_BINARY_DIR}/kernel_golden.regen.txt)
add_test(NAME manifest_1layer COMMAND vk_dispatch manifest 1 ${CMAKE_CURRENT_BINARY_DIR}/graph_manifest_1layer.json ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
add_test(NAME manifest_2layers COMMAND vk_dispatch manifest 2 ${CMAKE_CURRENT_BINARY_DIR}/graph_manifest_2layers.json ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
add_test(NAME manifest_1layer_reproducible
  COMMAND ${CMAKE_COMMAND} -E compare_files ${CMAKE_CURRENT_SOURCE_DIR}/manifests/graph_manifest_1layer.json ${CMAKE_CURRENT_BINARY_DIR}/graph_manifest_1layer.regen.json)
add_test(NAME manifest_2layers_reproducible
  COMMAND ${CMAKE_COMMAND} -E compare_files ${CMAKE_CURRENT_SOURCE_DIR}/manifests/graph_manifest_2layers.json ${CMAKE_CURRENT_BINARY_DIR}/graph_manifest_2layers.regen.json)
set_tests_properties(manifest_1layer manifest_2layers PROPERTIES PASS_REGULAR_EXPRESSION "MANIFEST OK")
foreach(shader ${ALL_SHADERS})
  add_test(NAME spirv_val_${shader} COMMAND ${SPIRV_VAL} --target-env vulkan1.2 ${CMAKE_CURRENT_BINARY_DIR}/${shader}.spv)
  add_test(NAME reflect_${shader} COMMAND check_reflection ${CMAKE_CURRENT_BINARY_DIR}/${shader}.spv ${shader})
endforeach()
add_test(NAME dispatch_list COMMAND vk_dispatch list)
set_tests_properties(dispatch_list PROPERTIES PASS_REGULAR_EXPRESSION "DEVICE:|BLOCKED:")
add_test(NAME dispatch_caps COMMAND vk_dispatch caps)
set_tests_properties(dispatch_caps PROPERTIES PASS_REGULAR_EXPRESSION "CAPS" SKIP_REGULAR_EXPRESSION "BLOCKED")
add_test(NAME dispatch_graph COMMAND vk_dispatch graph 1)
set_tests_properties(dispatch_graph PROPERTIES PASS_REGULAR_EXPRESSION "GRAPH VALID")
add_test(NAME dispatch_graph_2layers COMMAND vk_dispatch graph 2)
set_tests_properties(dispatch_graph_2layers PROPERTIES PASS_REGULAR_EXPRESSION "GRAPH VALID")
add_test(NAME dispatch_embedding COMMAND vk_dispatch embedding ${CMAKE_CURRENT_BINARY_DIR}/embedding.spv)
add_test(NAME dispatch_matvec COMMAND vk_dispatch matvec ${CMAKE_CURRENT_BINARY_DIR}/int4_matvec.spv)
set_tests_properties(dispatch_embedding dispatch_matvec PROPERTIES
  PASS_REGULAR_EXPRESSION "PASS" SKIP_REGULAR_EXPRESSION "BLOCKED")
