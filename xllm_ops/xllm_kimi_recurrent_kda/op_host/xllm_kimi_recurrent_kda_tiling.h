/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file xllm_kimi_recurrent_kda_tiling.h
 * \brief
 */
#ifndef __OP_HOST_RECURRENT_KDA_TILING_H__
#define __OP_HOST_RECURRENT_KDA_TILING_H__
#include <tiling/tiling_api.h>

#include "err/ops_err.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "xllm_kimi_recurrent_kda_tiling_data.h"
#include "xllm_kimi_recurrent_kda_tiling_processor.h"

namespace optiling {
using namespace XllmKimiRecurrentKda;

struct XllmKimiRecurrentKdaCompileInfo {
  uint64_t aivNum{0UL};
  uint64_t ubSize{0UL};
};

struct XllmKimiRecurrentKdaInfo {
 public:
  int64_t usedCoreNum = 0;
  const char* opName = "XllmKimiRecurrentKda";
};

class XllmKimiRecurrentKdaTiling : public Ops::Xllm::OpTiling::TilingBaseClass {
 public:
  explicit XllmKimiRecurrentKdaTiling(gert::TilingContext* context)
      : Ops::Xllm::OpTiling::TilingBaseClass(context) {
    InitCompileInfo();
  };
  ~XllmKimiRecurrentKdaTiling() override = default;

 protected:
  bool IsCapable() override { return true; }
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoLibApiTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus GetWorkspaceSize() override;
  ge::graphStatus PostTiling() override;

 protected:
  void InitCompileInfo();
  void PrintTilingData();
  XllmKimiRecurrentKdaTilingContext BuildProcessorContext() const;

  ge::graphStatus CheckContext();
  ge::graphStatus AnalyzeDtype();
  ge::graphStatus AnalyzeShapes();
  ge::graphStatus CalUbSize();
  ge::graphStatus GetAttrsInfo();
  ge::graphStatus GetOptionalInput();
  ge::graphStatus AnalyzeFormat();
  bool CheckFormat(ge::Format format, const std::string& Desc);

  XllmKimiRecurrentKdaCompileInfo compileInfo_;
  XllmKimiRecurrentKdaTilingData tilingData_;
  XllmKimiRecurrentKdaInfo inputParams_;
};

}  // namespace optiling
#endif  // __OP_HOST_RECURRENT_KDA_TILING_H__
