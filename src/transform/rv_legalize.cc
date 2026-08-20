// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../op/op.h"
#include "rv_ir.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

struct OperandSchema {
  int argument;
  const char *role;
};

const std::unordered_map<std::string, std::vector<OperandSchema>> &Schemas() {
  static const std::unordered_map<std::string, std::vector<OperandSchema>> schemas = {
      {"ppl.copy", {{1, "src"}, {2, "dst"}}},
      {"ppl.fill", {{1, "dst"}}},
      {"ppl.gemm", {{1, "lhs"}, {2, "rhs"}, {3, "dst"}}},
      {"ppl.sub", {{1, "dst"}, {2, "lhs"}, {3, "rhs"}}},
      {"ppl.mul", {{1, "dst"}, {2, "lhs"}, {3, "rhs"}}},
      {"ppl.add", {{1, "dst"}, {2, "lhs"}, {3, "rhs"}}},
      {"ppl.div", {{1, "dst"}, {2, "lhs"}, {3, "rhs"}}},
      {"ppl.mul_C", {{1, "dst"}, {2, "src"}}},
      {"ppl.add_C", {{1, "dst"}, {2, "src"}}},
      {"ppl.exp", {{1, "dst"}, {2, "work0"}, {3, "work1"},
                   {4, "coeff"}, {5, "table"}}},
      {"ppl.sigmoid", {{1, "dst"}, {2, "src"}, {3, "work0"},
                       {4, "work1"}, {5, "coeff"}, {6, "table"}}},
      {"ppl.rsqrt", {{1, "dst"}, {2, "src"}}},
      {"ppl.reduce_sum", {{1, "src"}, {2, "dst"}, {3, "tmp"}}},
      {"ppl.reduce_max", {{1, "src"}, {2, "dst"}, {3, "tmp"}}},
      {"ppl.gather", {{1, "dst"}, {2, "param"}, {3, "index"}}},
      {"ppl.topk", {{1, "dst_data"}, {2, "dst_index"}, {3, "src"}}},
      {"ppl.rope_add", {{1, "dst"}, {2, "even_lhs"}, {3, "even_rhs"},
                        {4, "odd_lhs"}, {5, "odd_rhs"}}},
  };
  return schemas;
}

enum class AccessKind : int { kRead = 0, kWrite = 1, kReadWrite = 2, kConservative = 3 };

AccessKind AccessForRole(const std::string &role) {
  if (role == "dst" || role == "dst_data" || role == "dst_index")
    return AccessKind::kWrite;
  if (role == "tmp")
    return AccessKind::kReadWrite;
  if (role.rfind("work", 0) == 0 || role == "coeff" || role == "table")
    return AccessKind::kConservative;
  return AccessKind::kRead;
}

bool HasScalarRegister(const std::string &name) {
  return name == "ppl.fill" || name == "ppl.mul_C" || name == "ppl.add_C";
}

int ScalarArgument(const std::string &name) {
  if (name == "ppl.fill")
    return 2;
  if (name == "ppl.mul_C" || name == "ppl.add_C")
    return 3;
  return -1;
}

Array<PrimExpr> Normalize4D(const Array<PrimExpr> &dims, bool local_layout) {
  ICHECK_GE(dims.size(), 1U);
  ICHECK_LE(dims.size(), 4U) << "RV tensor views support ranks 1 through 4";
  PrimExpr one = IntImm(DataType::Int(32), 1);
  Array<PrimExpr> result{one, one, one, one};
  if (dims.size() == 1) {
    result.Set(3, dims[0]);
  } else if (dims.size() == 2) {
    result.Set(1, dims[0]);
    result.Set(3, dims[1]);
  } else if (dims.size() == 3) {
    if (local_layout) {
      result.Set(0, dims[0]);
      result.Set(1, dims[1]);
      result.Set(3, dims[2]);
    } else {
      result.Set(1, dims[0]);
      result.Set(2, dims[1]);
      result.Set(3, dims[2]);
    }
  } else {
    result = dims;
  }
  return result;
}

Array<PrimExpr> CompactStrides(const Array<PrimExpr> &shape) {
  Array<PrimExpr> strides;
  PrimExpr product = IntImm(DataType::Int(32), 1);
  std::vector<PrimExpr> reversed;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    reversed.push_back(product);
    product = product * shape[i];
  }
  for (auto it = reversed.rbegin(); it != reversed.rend(); ++it)
    strides.push_back(*it);
  return strides;
}

class BufferCollector : public StmtExprVisitor {
public:
  explicit BufferCollector(const BufferMap &buffers) {
    for (const auto &entry : buffers)
      buffers_.Set(entry.second->data, entry.second);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    buffers_.Set(op->buffer->data, op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode *op) final {
    // Lowered T.alloc_buffer nodes are represented by Allocate rather than
    // DeclBuffer.  Preserve their pointer-typed Var (and therefore its storage
    // scope) while rebuilding the small amount of Buffer metadata required by
    // RV operand legalization.
    Buffer buffer(op->buffer_var, op->dtype, op->extents, {}, 0,
                  op->buffer_var->name_hint, 0, 0, BufferType::kDefault);
    buffers_.Set(op->buffer_var, buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers)
      buffers_.Set(buffer->data, buffer);
    for (const MatchBufferRegion &match : op->match_buffers)
      buffers_.Set(match->buffer->data, match->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  const BufferMap &buffers() const { return buffers_; }

private:
  BufferMap buffers_;
};

class RVLegalizer : public StmtExprMutator {
public:
  explicit RVLegalizer(BufferMap buffers) : buffers_(std::move(buffers)) {}

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !call->op.same_as(builtin::call_extern()) || call->args.empty())
      return StmtExprMutator::VisitStmt_(op);
    const auto *name_imm = call->args[0].as<StringImmNode>();
    std::string extern_name = name_imm ? std::string(name_imm->value) : "";
    if (!name_imm || extern_name.rfind("ppl.", 0) != 0 ||
        extern_name.rfind("ppl.rv.", 0) == 0)
      return StmtExprMutator::VisitStmt_(op);

    std::string name = name_imm->value;
    auto schema_it = Schemas().find(name);
    ICHECK(schema_it != Schemas().end())
        << "SG2260E RV legalization has no operand schema for " << name;

    Array<PrimExpr> rewritten_args = call->args;
    std::vector<std::pair<Var, PrimExpr>> bindings;
    for (const OperandSchema &operand : schema_it->second) {
      ICHECK_LT(static_cast<size_t>(operand.argument), call->args.size());
      const auto *access = call->args[operand.argument].as<CallNode>();
      ICHECK(access) << name << " operand " << operand.role
                     << " must be represented by tvm_access_ptr";
      Buffer buffer;
      Array<Range> ranges;
      bool full_region = true;
      if (access->op.same_as(RegionOp::Get())) {
        RegionOp region(access->args, buffers_);
        buffer = region.GetBuffer();
        ranges = region.GetRanges();
        full_region = region.IsFullRegion();
      } else {
        ICHECK(access->op.same_as(builtin::tvm_access_ptr()))
            << name << " operand " << operand.role
            << " must be a tile region or tvm_access_ptr";
        Var data = GetVarFromAccessPtr(call->args[operand.argument]);
        ICHECK(buffers_.count(data)) << "Cannot resolve RV operand buffer " << data;
        buffer = buffers_[data];
        for (const PrimExpr &extent : buffer->shape)
          ranges.push_back(Range::FromMinExtent(0, extent));
      }
      Var data = buffer->data;
      bool global = buffer.scope() == "global";
      rv::RegisterClass reg_class = global ? rv::RegisterClass::kGlobal
                                           : rv::RegisterClass::kTensor;
      rv::LayoutKind layout = global ? rv::LayoutKind::kContinuous
                                     : rv::LayoutKind::kHwAligned;
      if (!full_region && (name == "ppl.copy" || global))
        layout = rv::LayoutKind::kFree;
      int layout_key = static_cast<int>(layout);
      auto &registers = global ? gr_by_view_[data.get()] : tr_by_view_[data.get()];
      auto register_it = registers.find(layout_key);
      if (register_it == registers.end()) {
        int assigned = global ? next_gr_++ : next_tr_++;
        register_it = registers.emplace(layout_key, assigned).first;
      }
      int reg_num = register_it->second;

      Array<PrimExpr> dims;
      for (const Range &range : ranges)
        dims.push_back(range->extent);
      Array<PrimExpr> shape4 = Normalize4D(dims, !global);
      Array<PrimExpr> raw_strides = buffer->strides.empty()
                                           ? CompactStrides(buffer->shape)
                                           : buffer->strides;
      Array<PrimExpr> stride4 = Normalize4D(raw_strides, !global);

      Array<PrimExpr> view_args{
          StringImm(rv::kTensorView), StringImm(buffer->name),
          StringImm(operand.role), Integer(static_cast<int>(reg_class)),
          Integer(reg_num), Integer(static_cast<int>(layout)),
          Integer(global ? static_cast<int>(rv::MemoryScope::kGlobal)
                         : static_cast<int>(rv::MemoryScope::kLocal)),
          Integer(buffer->dtype.code()), Integer(buffer->dtype.bits()),
          Integer(buffer->dtype.lanes()),
          Integer(static_cast<int>(AccessForRole(operand.role))),
          call->args[operand.argument]};
      for (const PrimExpr &dim : shape4)
        view_args.push_back(dim);
      for (const PrimExpr &stride : stride4)
        view_args.push_back(stride);

      Var view(buffer->name + "_rv_" + operand.role + "_" +
                   std::to_string(operation_index_),
               DataType::Handle());
      bindings.emplace_back(
          view, Call(DataType::Handle(), builtin::call_extern(), view_args));
      rewritten_args.Set(operand.argument, view);
    }

    if (HasScalarRegister(name)) {
      int scalar_arg = ScalarArgument(name);
      ICHECK_LT(static_cast<size_t>(scalar_arg), call->args.size());
      PrimExpr scalar_value = call->args[scalar_arg];
      Array<PrimExpr> scalar_args{StringImm(rv::kScalar), Integer(next_cr_++),
                                  Integer(scalar_value.dtype().code()),
                                  Integer(scalar_value.dtype().bits()),
                                  Integer(scalar_value.dtype().lanes()),
                                  scalar_value};
      Var scalar("rv_cr_" + std::to_string(operation_index_), scalar_value.dtype());
      bindings.emplace_back(
          scalar, Call(scalar_value.dtype(), builtin::call_extern(), scalar_args));
      rewritten_args.Set(scalar_arg, scalar);
    }

    rewritten_args.Set(0, StringImm("ppl.rv." + name.substr(4)));
    Stmt body = Evaluate(Call(call->dtype, call->op, rewritten_args, call->span));
    for (auto it = bindings.rbegin(); it != bindings.rend(); ++it)
      body = LetStmt(it->first, it->second, body);
    ++operation_index_;
    return body;
  }

  int operation_count() const { return operation_index_; }

private:
  BufferMap buffers_;
  int operation_index_{0};
  int next_cr_{1};
  int next_tr_{8};
  int next_gr_{32};
  std::unordered_map<const VarNode *, std::unordered_map<int, int>> tr_by_view_;
  std::unordered_map<const VarNode *, std::unordered_map<int, int>> gr_by_view_;
};

PrimFunc LegalizeRV(PrimFunc function, String chip) {
  ICHECK_EQ(chip, "sg2260e") << "RV legalization currently supports sg2260e only";
  BufferCollector collector(function->buffer_map);
  collector(function->body);
  RVLegalizer legalizer(collector.buffers());
  Stmt body = legalizer(function->body);
  PrimFunc result = function;
  auto node = result.CopyOnWrite();
  node->body = body;
  auto attrs = node->attrs.CopyOnWrite();
  attrs->dict.Set(rv::kLegalizedAttr, Integer(1));
  attrs->dict.Set(rv::kSchemaVersionAttr, Integer(1));
  attrs->dict.Set(rv::kChipAttr, chip);
  attrs->dict.Set(rv::kDeviceModeAttr, String("rv"));
  attrs->dict.Set("tir.tpu.rv.operation_count",
                  Integer(legalizer.operation_count()));
  return result;
}

} // namespace

tvm::transform::Pass RVLegalizeAndAllocateRegisters(String chip) {
  using namespace tir::transform;
  auto pass_func = [chip](PrimFunc function, IRModule, PassContext) {
    return LegalizeRV(std::move(function), chip);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RVLegalizeAndAllocateRegisters", {});
}

TVM_REGISTER_GLOBAL("tl.transform.RVLegalizeAndAllocateRegisters")
    .set_body_typed(RVLegalizeAndAllocateRegisters);

} // namespace tl
} // namespace tvm
