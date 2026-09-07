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
#include "tpu_target_info.h"

namespace tvm {
namespace codegen {

// Return source code for the complete chip/programming-model Target identity.
// Host runtime and PPL SDK selection remain outside this codegen boundary.
std::string BuildTileLangTPU(IRModule mod, Target target) {
  bool output_ssa = false;
  tl::tpu::TargetSelection selection =
      tl::tpu::ResolveTarget(target, "TileLang TPU codegen");
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
    Optional<Target> function_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (function_target.defined()) {
      tl::tpu::TargetSelection function_selection = tl::tpu::ResolveTarget(
          function_target.value(), "TileLang TPU codegen PrimFunc");
      tl::tpu::CheckSameSelection(function_selection, selection,
                                  "TileLang TPU codegen");
    }
    cg.AddFunction(f);
  }

  return cg.Finish();
}

TVM_REGISTER_GLOBAL("target.build.tilelang_tpu")
    .set_body_typed(BuildTileLangTPU);

} // namespace codegen
} // namespace tvm
