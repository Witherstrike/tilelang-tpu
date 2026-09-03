/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "codegen_tpu_common.h"

#include <cctype>

namespace tvm {
namespace codegen {

namespace {

bool IsSupportedTPUChip(const std::string& chip) {
  return chip == "bm1690" || chip == "sg2260e";
}

bool SupportsProgrammingModel(const std::string& chip, const std::string& programming_model) {
  return programming_model == "tpukernel" ||
         (chip == "sg2260e" && programming_model == "rv");
}

bool LooksLikeTPUChipName(const std::string& value) {
  return value.size() >= 3 && (value.rfind("bm", 0) == 0 || value.rfind("sg", 0) == 0) &&
         std::isdigit(static_cast<unsigned char>(value[2]));
}

std::string TargetStringOrEmpty(const Optional<String>& value) {
  if (!value.defined() || value.value() == "unknown") {
    return "";
  }
  return value.value();
}

std::string LowerASCII(std::string value) {
  for (char& character : value) {
    character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  }
  return value;
}

struct TPUTargetSelection {
  std::string chip;
  std::string programming_model;
};

TPUTargetSelection GetTPUTargetSelection(const Target& target) {
  ICHECK_EQ(target->kind->name, "tpu")
      << "TileLang TPU codegen requires a TPU Target";
  const std::string mcpu = LowerASCII(TargetStringOrEmpty(target->GetAttr<String>("mcpu")));
  const std::string legacy_model =
      LowerASCII(TargetStringOrEmpty(target->GetAttr<String>("model")));
  const bool model_is_chip = IsSupportedTPUChip(legacy_model);

  // ``model`` is normally generic workload metadata, but a chip-looking
  // value is an obsolete device selector.  Reject an unknown SKU even when a
  // valid -mcpu is present; otherwise a typo would be silently retained.
  const bool unknown_chip_like_model =
      !legacy_model.empty() && LooksLikeTPUChipName(legacy_model) && !model_is_chip;
  ICHECK(!unknown_chip_like_model)
      << "Unsupported TPU chip in legacy target model attribute: model=" << legacy_model;

  if (!mcpu.empty()) {
    ICHECK(IsSupportedTPUChip(mcpu))
        << "TileLang TPU codegen requires a supported target chip via "
        << "tpu -mcpu=<bm1690|sg2260e>; got " << mcpu;
    ICHECK(!model_is_chip || legacy_model == mcpu)
        << "Conflicting TPU target attributes: mcpu=" << mcpu
        << ", model=" << legacy_model;
  }

  const std::string chip = !mcpu.empty() ? mcpu : (model_is_chip ? legacy_model : "");
  ICHECK(!chip.empty())
      << "TileLang TPU codegen requires a supported target chip via "
      << "tpu -mcpu=<bm1690|sg2260e>; got "
      << (chip.empty() ? "no chip" : chip);

  const std::string programming_model =
      TargetStringOrEmpty(target->GetAttr<String>("tpu-programming-model"));
  ICHECK(programming_model == "tpukernel" || programming_model == "rv")
      << "TileLang TPU codegen requires a normalized target "
      << "tpu-programming-model=<tpukernel|rv>; got "
      << (programming_model.empty() ? "no programming model" : programming_model);
  ICHECK(SupportsProgrammingModel(chip, programming_model))
      << "TPU target chip " << chip << " does not support "
      << "tpu-programming-model=" << programming_model;
  return {chip, programming_model};
}

}  // namespace

// Return source code. The Target carries the same chip selected by the PPL
// layout resolver, so codegen is no longer implicitly BM1690-only.
std::string BuildTileLangTPU(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  TPUTargetSelection selection = GetTPUTargetSelection(target);
  CodeGenTileLangTPU cg(selection.chip, selection.programming_model);
  cg.Init(output_ssa);

  ICHECK_EQ(mod->functions.size(), 1U)
      << "TileLang TPU runtime codegen currently accepts exactly one PrimFunc "
         "per module because it emits one registered main_kernel entry; inline "
         "or split helper PrimFuncs before this boundary";

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "TileLang TPU codegen can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    // auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    // ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();

  return code;
  // return runtime::CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.tilelang_tpu")
    .set_body_typed(BuildTileLangTPU);

// Compatibility key for callers compiled against the original TPU backend.
// Both names deliberately share one validated dispatcher.
TVM_REGISTER_GLOBAL("target.build.tilelang_ppl")
    .set_body_typed(BuildTileLangTPU);
// TVM_REGISTER_GLOBAL("target.build.tl_debug_codegen").set_body_typed(BuildTLDebug);

} // namespace codegen
} // namespace tvm
