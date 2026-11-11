# Vulkan/ — Phase B subsystem: INT4 oracle, Vulkan compute runtime,
# GLSL->SPIR-V shaders, verification. Independent of the Swift
# package (which Phase B deliberately does not modify).
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

add_library(vulkan_runtime src/vulkan_runtime.cpp)
target_include_directories(vulkan_runtime PUBLIC include PRIVATE ${VK_INCLUDE})
target_link_libraries(vulkan_runtime PUBLIC ${CMAKE_DL_LIBS})

add_executable(vk_dispatch tools/vk_dispatch.cpp)
target_link_libraries(vk_dispatch PRIVATE vulkan_runtime int4_oracle)
target_include_directories(vk_dispatch PRIVATE tests)

add_executable(gen_golden tests/gen_golden.cpp)
target_link_libraries(gen_golden PRIVATE int4_oracle)
target_include_directories(gen_golden PRIVATE tests)

add_executable(test_int4_oracle tests/test_int4_oracle.cpp)
target_link_libraries(test_int4_oracle PRIVATE int4_oracle)
target_include_directories(test_int4_oracle PRIVATE tests)

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
foreach(shader embedding int4_matvec)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${shader}.spv
    COMMAND ${GLSLANG} -V --target-env vulkan1.2
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp
            -o ${CMAKE_CURRENT_BINARY_DIR}/${shader}.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp
    COMMENT "glslang ${shader}.comp -> ${shader}.spv")
  add_custom_target(spv_${shader} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${shader}.spv)
endforeach()

# --- golden vectors (checked-in copy is the one tests consume) ---
set(GOLDEN ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/int4_golden.txt)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/int4_golden.regen.txt
  COMMAND gen_golden ${CMAKE_CURRENT_BINARY_DIR}/int4_golden.regen.txt
  DEPENDS gen_golden)
add_custom_target(golden_regen ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/int4_golden.regen.txt)

# --- tests ---
add_test(NAME int4_oracle COMMAND test_int4_oracle ${GOLDEN})
add_test(NAME golden_reproducible
  COMMAND ${CMAKE_COMMAND} -E compare_files ${GOLDEN} ${CMAKE_CURRENT_BINARY_DIR}/int4_golden.regen.txt)
add_test(NAME spirv_val_embedding COMMAND ${SPIRV_VAL} --target-env vulkan1.2 ${CMAKE_CURRENT_BINARY_DIR}/embedding.spv)
add_test(NAME spirv_val_matvec COMMAND ${SPIRV_VAL} --target-env vulkan1.2 ${CMAKE_CURRENT_BINARY_DIR}/int4_matvec.spv)
add_test(NAME reflect_embedding COMMAND check_reflection ${CMAKE_CURRENT_BINARY_DIR}/embedding.spv embedding)
add_test(NAME reflect_matvec COMMAND check_reflection ${CMAKE_CURRENT_BINARY_DIR}/int4_matvec.spv matvec)
add_test(NAME dispatch_list COMMAND vk_dispatch list)
set_tests_properties(dispatch_list PROPERTIES PASS_REGULAR_EXPRESSION "DEVICE:|BLOCKED:")
add_test(NAME dispatch_embedding COMMAND vk_dispatch embedding ${CMAKE_CURRENT_BINARY_DIR}/embedding.spv)
add_test(NAME dispatch_matvec COMMAND vk_dispatch matvec ${CMAKE_CURRENT_BINARY_DIR}/int4_matvec.spv)
set_tests_properties(dispatch_embedding dispatch_matvec PROPERTIES
  PASS_REGULAR_EXPRESSION "PASS" SKIP_REGULAR_EXPRESSION "BLOCKED")
