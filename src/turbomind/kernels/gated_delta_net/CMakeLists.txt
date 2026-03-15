# Copyright (c) OpenMMLab. All rights reserved.

add_library(causal_conv1d STATIC causal_conv1d.cu)
set_property(TARGET causal_conv1d PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET causal_conv1d PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(delta_rule STATIC delta_rule.cu)
set_property(TARGET delta_rule PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET delta_rule PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(gated_rmsnorm STATIC gated_rmsnorm.cu)
set_property(TARGET gated_rmsnorm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET gated_rmsnorm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(gdn_utils STATIC gdn_utils.cu)
set_property(TARGET gdn_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET gdn_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Standalone correctness test (not built by default)
add_executable(test_causal_conv1d test_causal_conv1d.cu)
target_link_libraries(test_causal_conv1d PRIVATE causal_conv1d)
set_target_properties(test_causal_conv1d PROPERTIES EXCLUDE_FROM_ALL TRUE)

# Standalone prefill micro-benchmark (not built by default)
add_executable(test_delta_rule_prefill test_delta_rule_prefill.cu)
target_link_libraries(test_delta_rule_prefill PRIVATE delta_rule)
set_target_properties(test_delta_rule_prefill PROPERTIES EXCLUDE_FROM_ALL TRUE)
