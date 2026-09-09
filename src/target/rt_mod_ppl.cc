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

#include "codegen_ppl.h"
#include <tvm/target/target_kind.h>

namespace tvm {
namespace codegen {

// A CPU-only TVM build may not contain the downstream TPU target patch.
// Register the source backend here without overriding patched TVM builds.
static const bool tpu_target_registered = []() {
  if (!TargetKind::Get("tpu").defined()) {
    TargetKindRegEntry::RegisterOrGet("tpu")
        .set_name()
        .set_default_device_type(kDLCPU)
        .add_attr_option<Array<String>>("keys")
        .add_attr_option<String>("tag")
        .add_attr_option<String>("device")
        .add_attr_option<String>("model")
        .add_attr_option<Array<String>>("libs")
        .add_attr_option<Target>("host")
        .add_attr_option<Integer>("from_device")
        .add_attr_option<Integer>("target_device_type")
        .set_default_keys({"cpu"})
        .add_attr_option<String>("mcpu")
        .add_attr_option<String>("march")
        .add_attr_option<Integer>("workspace-byte-alignment")
        .add_attr_option<Integer>("constants-byte-alignment");
  }
  return true;
}();

// return source code.
std::string BuildTileLangPPL(IRModule mod) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenTileLangPPL cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangPPL: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    // auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    // ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();

  return code;
  // return runtime::CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.tilelang_ppl")
    .set_body_typed(BuildTileLangPPL);
// TVM_REGISTER_GLOBAL("target.build.tl_debug_codegen").set_body_typed(BuildTLDebug);

} // namespace codegen
} // namespace tvm
