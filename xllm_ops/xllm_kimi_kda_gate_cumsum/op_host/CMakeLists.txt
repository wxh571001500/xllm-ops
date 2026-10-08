# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE xllm_kimi_kda_gate_cumsum_def.cpp)
endif()
add_modules_sources(OPTYPE xllm_kimi_kda_gate_cumsum ACLNNTYPE aclnn_exclude)
add_ops_compile_options(OP_NAME XllmKimiKdaGateCumsum
    OPTIONS -I${CMAKE_SOURCE_DIR} --cce-auto-sync=off -Wno-deprecated-declarations)

target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(op_host_aclnnExc PRIVATE ${CMAKE_SOURCE_DIR})
