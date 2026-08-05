// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_TARGET_CODEGEN_PPL_RV_H_
#define TVM_TL_TARGET_CODEGEN_PPL_RV_H_

#include <tvm/tir/function.h>

#include <string>
#include <unordered_map>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

/*! \brief C emitter for the explicit SG2260E ppl.rv instruction IR. */
class CodeGenTileLangPPLRV final : public CodeGenC {
public:
  CodeGenTileLangPPLRV();
  void AddFunction(const PrimFunc &f);
  std::string Finish();

private:
  struct TensorView {
    int register_class;
    int register_number;
    int layout;
    int memory_scope;
    int dtype_code;
    int dtype_bits;
    int dtype_lanes;
    int access;
    std::string buffer_name;
    PrimExpr address;
    Array<PrimExpr> shape;
    Array<PrimExpr> stride;
  };

  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const LetStmtNode *op) final;
  void VisitStmt_(const EvaluateNode *op) final;
  void EmitTensorView(const TensorView &view);
  TensorView ParseTensorView(const CallNode *call) const;
  const TensorView &GetTensorView(const PrimExpr &expr,
                                  const std::string &operation) const;
  void ValidateSameTensor(const TensorView &lhs, const TensorView &rhs,
                          const std::string &operation) const;

  std::unordered_map<const VarNode *, TensorView> tensor_views_;
  std::unordered_map<int, std::string> configured_registers_;
  std::unordered_map<std::string, int64_t> local_addresses_;
  std::unordered_map<std::string, std::string> global_addresses_;
};

std::string BuildTileLangPPLRV(IRModule mod);

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_PPL_RV_H_
