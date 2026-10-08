/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of the BSD 3-Clause License (the "License"). Please
 * refer to the License for details. You may not use this file except in
 * compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS,
 * WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT
 * LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
 * PURPOSE.
 */

/*!
 * \file xllm_kimi_chunk_gated_delta_rule_fwd_h.cpp
 * \brief
 */

#include "arch22/gemm/kernel/gdn_fwd_h_kernel.hpp"
#include "lib/matmul_intf.h"

using namespace Catlass;

namespace GDN {

template <typename InputT,
          typename GT,
          typename StateT,
          typename WorkspaceT,
          typename TileShapes,
          bool kGated>
__aicore__ inline void XllmKimiChunkGatedDeltaRuleFwdHKernelImpl(
    GM_ADDR k,
    GM_ADDR w,
    GM_ADDR u,
    GM_ADDR g,
    GM_ADDR gk,
    GM_ADDR inital_state,
    GM_ADDR cu_seqlens,
    GM_ADDR chunk_indices,
    GM_ADDR h,
    GM_ADDR v_new,
    GM_ADDR final_state,
    GM_ADDR tiling,
    GM_ADDR user) {
  using GDNFwdHKernel = Catlass::Gemm::Kernel::
      GDNFwdHKernel<InputT, GT, StateT, WorkspaceT, TileShapes, kGated>;
  GDNFwdHKernel gdnFwdH;
  gdnFwdH.Init(k,
               w,
               u,
               g,
               gk,
               inital_state,
               cu_seqlens,
               chunk_indices,
               h,
               v_new,
               final_state,
               tiling,
               user);
  gdnFwdH.Process();
}

template <typename TileShapes>
__aicore__ inline void XllmKimiChunkGatedDeltaRuleFwdHDispatch(
    GM_ADDR k,
    GM_ADDR w,
    GM_ADDR u,
    GM_ADDR g,
    GM_ADDR gk,
    GM_ADDR inital_state,
    GM_ADDR cu_seqlens,
    GM_ADDR chunk_indices,
    GM_ADDR h,
    GM_ADDR v_new,
    GM_ADDR final_state,
    GM_ADDR tiling,
    GM_ADDR user) {
  __gm__ XllmKimiChunkGatedDeltaRuleFwdHTilingData* __restrict gdnFwdHTilingData =
      reinterpret_cast<
          __gm__ XllmKimiChunkGatedDeltaRuleFwdHTilingData* __restrict>(tiling);
  using WorkspaceT = float;
  bool useGk = gdnFwdHTilingData->useGk;
  // dtype: 0 - fp16, 1 - bf16, 2 - fp32
  if (gdnFwdHTilingData->dataType == 1) {
    if (gdnFwdHTilingData->stateDataType == 2) {
      if (gdnFwdHTilingData->gDataType == 2) {
        if (useGk) {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t,
                                                    float,
                                                    float,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    true>(k,
                                                          w,
                                                          u,
                                                          g,
                                                          gk,
                                                          inital_state,
                                                          cu_seqlens,
                                                          chunk_indices,
                                                          h,
                                                          v_new,
                                                          final_state,
                                                          tiling,
                                                          user);
        } else {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t,
                                                    float,
                                                    float,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    false>(k,
                                                           w,
                                                           u,
                                                           g,
                                                           gk,
                                                           inital_state,
                                                           cu_seqlens,
                                                           chunk_indices,
                                                           h,
                                                           v_new,
                                                           final_state,
                                                           tiling,
                                                           user);
        }
      } else {
        if (useGk) {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t,
                                                    bfloat16_t,
                                                    float,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    true>(k,
                                                          w,
                                                          u,
                                                          g,
                                                          gk,
                                                          inital_state,
                                                          cu_seqlens,
                                                          chunk_indices,
                                                          h,
                                                          v_new,
                                                          final_state,
                                                          tiling,
                                                          user);
        } else {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t,
                                                    bfloat16_t,
                                                    float,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    false>(k,
                                                           w,
                                                           u,
                                                           g,
                                                           gk,
                                                           inital_state,
                                                           cu_seqlens,
                                                           chunk_indices,
                                                           h,
                                                           v_new,
                                                           final_state,
                                                           tiling,
                                                           user);
        }
      }
    } else {
      if (gdnFwdHTilingData->gDataType == 2) {
        if (useGk) {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t,
                                                    float,
                                                    bfloat16_t,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    true>(k,
                                                          w,
                                                          u,
                                                          g,
                                                          gk,
                                                          inital_state,
                                                          cu_seqlens,
                                                          chunk_indices,
                                                          h,
                                                          v_new,
                                                          final_state,
                                                          tiling,
                                                          user);
        } else {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t,
                                                    float,
                                                    bfloat16_t,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    false>(k,
                                                           w,
                                                           u,
                                                           g,
                                                           gk,
                                                           inital_state,
                                                           cu_seqlens,
                                                           chunk_indices,
                                                           h,
                                                           v_new,
                                                           final_state,
                                                           tiling,
                                                           user);
        }
      } else {
        if (useGk) {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t,
                                                    bfloat16_t,
                                                    bfloat16_t,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    true>(k,
                                                          w,
                                                          u,
                                                          g,
                                                          gk,
                                                          inital_state,
                                                          cu_seqlens,
                                                          chunk_indices,
                                                          h,
                                                          v_new,
                                                          final_state,
                                                          tiling,
                                                          user);
        } else {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t,
                                                    bfloat16_t,
                                                    bfloat16_t,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    false>(k,
                                                           w,
                                                           u,
                                                           g,
                                                           gk,
                                                           inital_state,
                                                           cu_seqlens,
                                                           chunk_indices,
                                                           h,
                                                           v_new,
                                                           final_state,
                                                           tiling,
                                                           user);
        }
      }
    }
  } else {
    if (gdnFwdHTilingData->stateDataType == 2) {
      if (gdnFwdHTilingData->gDataType == 2) {
        if (useGk) {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<half,
                                                    float,
                                                    float,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    true>(k,
                                                          w,
                                                          u,
                                                          g,
                                                          gk,
                                                          inital_state,
                                                          cu_seqlens,
                                                          chunk_indices,
                                                          h,
                                                          v_new,
                                                          final_state,
                                                          tiling,
                                                          user);
        } else {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<half,
                                                    float,
                                                    float,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    false>(k,
                                                           w,
                                                           u,
                                                           g,
                                                           gk,
                                                           inital_state,
                                                           cu_seqlens,
                                                           chunk_indices,
                                                           h,
                                                           v_new,
                                                           final_state,
                                                           tiling,
                                                           user);
        }
      } else {
        if (useGk) {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<half,
                                                    half,
                                                    float,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    true>(k,
                                                          w,
                                                          u,
                                                          g,
                                                          gk,
                                                          inital_state,
                                                          cu_seqlens,
                                                          chunk_indices,
                                                          h,
                                                          v_new,
                                                          final_state,
                                                          tiling,
                                                          user);
        } else {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<half,
                                                    half,
                                                    float,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    false>(k,
                                                           w,
                                                           u,
                                                           g,
                                                           gk,
                                                           inital_state,
                                                           cu_seqlens,
                                                           chunk_indices,
                                                           h,
                                                           v_new,
                                                           final_state,
                                                           tiling,
                                                           user);
        }
      }
    } else {
      if (gdnFwdHTilingData->gDataType == 2) {
        if (useGk) {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<half,
                                                    float,
                                                    half,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    true>(k,
                                                          w,
                                                          u,
                                                          g,
                                                          gk,
                                                          inital_state,
                                                          cu_seqlens,
                                                          chunk_indices,
                                                          h,
                                                          v_new,
                                                          final_state,
                                                          tiling,
                                                          user);
        } else {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<half,
                                                    float,
                                                    half,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    false>(k,
                                                           w,
                                                           u,
                                                           g,
                                                           gk,
                                                           inital_state,
                                                           cu_seqlens,
                                                           chunk_indices,
                                                           h,
                                                           v_new,
                                                           final_state,
                                                           tiling,
                                                           user);
        }
      } else {
        if (useGk) {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<half,
                                                    half,
                                                    half,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    true>(k,
                                                          w,
                                                          u,
                                                          g,
                                                          gk,
                                                          inital_state,
                                                          cu_seqlens,
                                                          chunk_indices,
                                                          h,
                                                          v_new,
                                                          final_state,
                                                          tiling,
                                                          user);
        } else {
          XllmKimiChunkGatedDeltaRuleFwdHKernelImpl<half,
                                                    half,
                                                    half,
                                                    WorkspaceT,
                                                    TileShapes,
                                                    false>(k,
                                                           w,
                                                           u,
                                                           g,
                                                           gk,
                                                           inital_state,
                                                           cu_seqlens,
                                                           chunk_indices,
                                                           h,
                                                           v_new,
                                                           final_state,
                                                           tiling,
                                                           user);
        }
      }
    }
  }
}

}  // namespace GDN

extern "C" __global__ __aicore__ void xllm_kimi_chunk_gated_delta_rule_fwd_h(
    GM_ADDR k,
    GM_ADDR w,
    GM_ADDR u,
    GM_ADDR g,
    GM_ADDR gk,
    GM_ADDR inital_state,
    GM_ADDR cu_seqlens,
    GM_ADDR chunk_indices,
    GM_ADDR h,
    GM_ADDR v_new,
    GM_ADDR final_state,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  GM_ADDR user = AscendC::GetUserWorkspace(workspace);

  if (TILING_KEY_IS(1)) {
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    GDN::XllmKimiChunkGatedDeltaRuleFwdHDispatch<
        Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(k,
                                                     w,
                                                     u,
                                                     g,
                                                     gk,
                                                     inital_state,
                                                     cu_seqlens,
                                                     chunk_indices,
                                                     h,
                                                     v_new,
                                                     final_state,
                                                     tiling,
                                                     user);
  } else if (TILING_KEY_IS(2)) {
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    GDN::XllmKimiChunkGatedDeltaRuleFwdHDispatch<
        Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(k,
                                                     w,
                                                     u,
                                                     g,
                                                     gk,
                                                     inital_state,
                                                     cu_seqlens,
                                                     chunk_indices,
                                                     h,
                                                     v_new,
                                                     final_state,
                                                     tiling,
                                                     user);
  }
}
