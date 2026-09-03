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

/*!
 * \file target/codegen_tpu_common.h
 * \brief Shared TileLang TPU source generation and backend dispatch.
 */
#ifndef TVM_TL_TARGET_CODEGEN_TPU_COMMON_H_
#define TVM_TL_TARGET_CODEGEN_TPU_COMMON_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangTPU final : public CodeGenC {
public:
  CodeGenTileLangTPU(std::string target_chip, std::string target_programming_model);
  std::string Finish();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f, std::ostream &os) final;
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final; // NOLINT(*)
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) final;      // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  void BindThreadIndex(const IterVar &iv) final; // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, DataType from,
                         DataType target) final;
  // overload visitor
  void VisitExpr_(const RampNode *op, std::ostream &os) final; // NOLINT(*)
  // void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  //
  // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const LetStmtNode *op) final;
  void VisitExpr_(const FloorDivNode *op, std::ostream &os) final;
  void VisitExpr_(const FloorModNode *op, std::ostream &os) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const PrimFunc &f);

protected:
  virtual std::string GetBufferRef(DataType t, const BufferNode *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, String global_symbol,
                       const Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  // Backend-specific emission.  Parsing TIR operands and maintaining the
  // stable kernel ABI stay in this class; instruction selection lives in the
  // two sibling translation units named after their programming model.
  void EmitTPUKernelCopy(const std::string &src, bool src_is_global,
                         const std::string &src_dtype, const std::string &dst,
                         bool dst_is_global, const std::string &dst_dtype);
  void EmitRVCopy(const std::string &src, bool src_is_global,
                  const std::string &src_dtype, const std::string &dst,
                  bool dst_is_global, const std::string &dst_dtype);
  void EmitTPUKernelFill(const std::string &dst, DataType dtype, double value);
  void EmitRVFill(const std::string &dst, DataType dtype, double value);
  void EmitTPUKernelGemm(const std::string &a, const std::string &b,
                         const std::string &c, DataType a_dtype,
                         DataType b_dtype, DataType c_dtype, bool transpose_a,
                         bool transpose_b, bool accumulate, int64_t m,
                         int64_t n, int64_t k);
  void EmitRVGemm(const std::string &a, const std::string &b,
                  const std::string &c, DataType a_dtype, DataType b_dtype,
                  DataType c_dtype, bool transpose_a, bool transpose_b,
                  bool accumulate, int64_t m, int64_t n, int64_t k);
  void EmitTPUKernelElementwise(const std::string &instruction,
                                const std::string &dst,
                                const std::string &src0,
                                const std::string &src1,
                                const std::string &dtype,
                                const std::string &src1_stride);
  void EmitRVElementwise(const std::string &operation,
                         const std::string &dst, const std::string &src0,
                         const std::string &src1, DataType dst_dtype,
                         DataType src0_dtype, DataType src1_dtype,
                         const std::vector<int> &dst_shape,
                         const std::vector<int> &src0_shape,
                         const std::vector<int> &src1_shape);
  void EmitRVDescriptor(const std::string &tensor, int register_id,
                        bool is_global, const std::string &dtype,
                        bool hw_aligned);

  // Handle volatile loads
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *op,
                           std::ostream &os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangTPU *p);
  std::string AllocLocalVarID(const tir::VarNode *v);
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};
  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;
  const int lane_num = 64;
  std::unordered_map<const VarNode *, std::string> fragment_shapes;
  std::unordered_map<const VarNode *, std::string> fragment_layouts;
  std::unordered_map<std::string, std::string> parameter_map;
  std::unordered_map<std::string, std::vector<int>> buffer_shape;
  // Full normalized N/C/H/W shape for local tensors.  buffer_shape retains
  // the legacy C/W view used by older TPU-Kernel emitters; portable core ops
  // must consult this map so a non-trivial N or H dimension is never silently
  // flattened into an incompatible matrix/vector descriptor.
  std::unordered_map<std::string, std::vector<int>> buffer_shape4;
  std::unordered_map<std::string, std::vector<int>> buffer_stride;
  std::unordered_map<const VarNode *, std::vector<std::string>>
      local_buffer_name_map;

  std::unordered_map<const VarNode *, int> buffer_addrs_;

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangTPU *p);
  void PrintWmmaScope(const std::string &scope, DataType t,
                      const VarNode *variable, std::ostream &os);
  int32_t GetWmmaFragmentSize(const std::string &scope, const VarNode *variable,
                              int32_t size);
  int32_t gemm_idx_ = 0;
  // Module-level include/ABI requirements plus per-function ownership fences.
  bool uses_rvt_api_{false};
  bool uses_tpukernel_api_{false};
  bool uses_canonical_rv_{false};
  bool uses_opaque_raw_rvt_{false};
  int rvt_direct_call_count_{0};
  int tpukernel_extern_count_{0};
  int canonical_tpu_op_count_{0};
  std::string target_chip_;
  std::string target_programming_model_;

  DictAttrs f_attrs;
  std::vector<std::pair<tir::Var, Range>> loop_var_ranges_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_TPU_COMMON_H_
