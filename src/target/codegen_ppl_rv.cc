// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "codegen_ppl_rv.h"

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../op/op.h"
#include "../transform/rv_ir.h"

namespace tvm {
namespace codegen {

namespace {

int AsInt(const PrimExpr &expr, const char *field) {
  const auto *value = expr.as<IntImmNode>();
  ICHECK(value) << "SG2260E RV tensor view " << field << " must be constant";
  return static_cast<int>(value->value);
}

const char *LayoutName(int layout) {
  switch (static_cast<tl::rv::LayoutKind>(layout)) {
  case tl::rv::LayoutKind::kHwAligned:
    return "HW_ALIGN_LAYOUT";
  case tl::rv::LayoutKind::kContinuous:
    return "CONTINUOUS_LAYOUT";
  case tl::rv::LayoutKind::kRowAligned:
    return "ROW_ALIGN_LAYOUT";
  case tl::rv::LayoutKind::kFree:
    return "FREE_LAYOUT";
  case tl::rv::LayoutKind::kMatrix:
    return "MATRIX_LAYOUT";
  }
  LOG(FATAL) << "Unsupported SG2260E RV layout " << layout;
  return "";
}

const char *ElementWidth(int code, int bits) {
  if (code == DataType::kFloat && bits == 16)
    return "TEEW_E16";
  if (code == DataType::kFloat && bits == 32)
    return "TEEW_E32";
  if ((code == DataType::kInt || code == DataType::kUInt) && bits == 8)
    return "TEEW_E8";
  if ((code == DataType::kInt || code == DataType::kUInt) && bits == 16)
    return "TEEW_E16";
  if ((code == DataType::kInt || code == DataType::kUInt) && bits == 32)
    return "TEEW_E32";
  LOG(FATAL) << "Unsupported SG2260E RV dtype code=" << code << ", bits=" << bits;
  return "";
}

} // namespace

CodeGenTileLangPPLRV::CodeGenTileLangPPLRV() { restrict_keyword_ = ""; }

std::string CodeGenTileLangPPLRV::Finish() {
  decl_stream << "#include \"tpu_kernel.h\"\n"
              << "#include \"ppl_helper.h\"\n"
              << "#include \"atomic_def.h\"\n"
              << "#include \"rvt_api.h\"\n\n";
  return CodeGenC::Finish();
}

void CodeGenTileLangPPLRV::AddFunction(const PrimFunc &f) {
  ICHECK(f->HasNonzeroAttr(tl::rv::kLegalizedAttr))
      << "CodeGenTileLangPPLRV requires RV-legalized TIR";
  ICHECK_EQ(f->GetAttr<String>(tl::rv::kChipAttr).value_or(""), "sg2260e")
      << "CodeGenTileLangPPLRV supports SG2260E only";
  ICHECK_EQ(f->GetAttr<String>(tl::rv::kDeviceModeAttr).value_or(""), "rv")
      << "CodeGenTileLangPPLRV requires tir.tpu.device_mode=rv";
  ICHECK_EQ(f->GetAttr<Integer>(tl::rv::kSchemaVersionAttr).value_or(0), 1)
      << "CodeGenTileLangPPLRV supports RV schema version 1 only";
  auto symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(symbol.defined()) << "RV PrimFunc requires global_symbol";

  InitFuncState(f);
  tensor_views_.clear();
  configured_registers_.clear();
  local_addresses_.clear();
  global_addresses_.clear();
  for (const auto &attr : f->attrs->dict) {
    if (const auto *address = attr.second.as<IntImmNode>())
      local_addresses_[attr.first] = address->value;
  }

  std::string function_name = symbol.value();
  stream << "typedef struct {\n";
  for (const Var &param : f->params) {
    std::string field = f->buffer_map.count(param)
                            ? std::string(f->buffer_map[param]->name)
                            : std::string(param->name_hint);
    stream << "  ";
    if (f->buffer_map.count(param))
      stream << "global_addr_t";
    else
      PrintType(param.dtype(), stream);
    stream << " " << field << ";\n";
  }
  stream << "} tpu_kernel_api_" << function_name << "_t;\n";
  stream << "void " << function_name << "_inner(";
  std::vector<std::string> field_names;
  for (size_t i = 0; i < f->params.size(); ++i) {
    if (i)
      stream << ", ";
    const Var &param = f->params[i];
    std::string name = AllocVarID(param.get());
    if (f->buffer_map.count(param)) {
      field_names.push_back(f->buffer_map[param]->name);
      // TIR buffer data is commonly a distinct Var from the handle parameter.
      // Both denote the same global address in the emitted kernel ABI.
      var_idmap_[f->buffer_map[param]->data.get()] = name;
      global_addresses_[f->buffer_map[param]->name] = name;
      stream << "global_addr_t " << name;
    } else {
      field_names.push_back(param->name_hint);
      PrintType(param.dtype(), stream);
      stream << " " << name;
    }
  }
  stream << ") {\n";
  int function_scope = BeginScope();
  PrintIndent();
  stream << "rvt_cfg_lanemask(gdma_get_lane_mask());\n";
  PrintStmt(f->body);
  EndScope(function_scope);
  stream << "}\n";
  stream << "int " << function_name << "_entry(const void *args) {\n"
         << "  tpu_kernel_api_" << function_name
         << "_t *api = (tpu_kernel_api_" << function_name << "_t *)args;\n"
         << "  rvt_kernel_start();\n"
         << "  " << function_name << "_inner(";
  for (size_t i = 0; i < field_names.size(); ++i) {
    if (i)
      stream << ", ";
    stream << "api->" << field_names[i];
  }
  stream << ");\n  rvt_sync_i(0xdeadbeef, 0);\n  return 0;\n}\n";
}

void CodeGenTileLangPPLRV::VisitStmt_(const AllocateNode *op) {
  PrintStmt(op->body);
}

CodeGenTileLangPPLRV::TensorView
CodeGenTileLangPPLRV::ParseTensorView(const CallNode *call) const {
  ICHECK_EQ(call->args.size(), 20U)
      << "SG2260E RV tensor_view schema mismatch: expected 20 arguments";
  const auto *buffer_name = call->args[1].as<StringImmNode>();
  ICHECK(buffer_name) << "SG2260E RV tensor view buffer name must be constant";
  TensorView view{AsInt(call->args[3], "register class"),
                  AsInt(call->args[4], "register number"),
                  AsInt(call->args[5], "layout"),
                  AsInt(call->args[6], "memory scope"),
                  AsInt(call->args[7], "dtype code"),
                  AsInt(call->args[8], "dtype bits"),
                  AsInt(call->args[9], "dtype lanes"),
                  AsInt(call->args[10], "access"),
                  buffer_name->value,
                  call->args[11],
                  {call->args[12], call->args[13], call->args[14], call->args[15]},
                  {call->args[16], call->args[17], call->args[18], call->args[19]}};
  ICHECK_EQ(view.dtype_lanes, 1) << "SG2260E RV supports scalar element dtypes only";
  ICHECK_GE(view.access, 0);
  ICHECK_LE(view.access, 3) << "SG2260E RV tensor view has invalid access mode";
  for (const PrimExpr &dim : view.shape)
    ICHECK_GT(AsInt(dim, "shape"), 0) << "SG2260E RV shapes must be positive";
  for (const PrimExpr &stride : view.stride)
    ICHECK_GT(AsInt(stride, "stride"), 0) << "SG2260E RV strides must be positive";
  return view;
}

void CodeGenTileLangPPLRV::EmitTensorView(const TensorView &view) {
  bool global = view.register_class == static_cast<int>(tl::rv::RegisterClass::kGlobal);
  ICHECK(global || view.register_class == static_cast<int>(tl::rv::RegisterClass::kTensor))
      << "SG2260E RV tensor_view requires GR or TR register class";
  ICHECK_EQ(view.memory_scope,
            global ? static_cast<int>(tl::rv::MemoryScope::kGlobal)
                   : static_cast<int>(tl::rv::MemoryScope::kLocal))
      << "SG2260E RV register class and memory scope disagree for " << view.buffer_name;
  const auto *access = view.address.as<CallNode>();
  ICHECK(access) << "SG2260E RV tensor view " << view.buffer_name
                 << " has an unparseable address expression";
  std::string backing_address;
  PrimExpr element_offset = IntImm(DataType::Int(64), 0);
  if (access->op.same_as(tir::builtin::tvm_access_ptr())) {
    ICHECK_GE(access->args.size(), 5U)
        << "SG2260E RV tvm_access_ptr schema mismatch for " << view.buffer_name;
    if (global) {
      auto backing = global_addresses_.find(view.buffer_name);
      ICHECK(backing != global_addresses_.end())
          << "SG2260E RV global buffer " << view.buffer_name
          << " is not mapped to a PrimFunc parameter";
      backing_address = backing->second;
    }
    element_offset = access->args[2];
  } else if (access->op.same_as(tl::RegionOp::Get())) {
    ICHECK_GE(access->args.size(), 3U)
        << "SG2260E RV tile region schema mismatch for " << view.buffer_name;
    const auto *load = access->args[0].as<BufferLoadNode>();
    ICHECK(load) << "SG2260E RV tile region must begin with BufferLoad for "
                 << view.buffer_name;
    ICHECK_EQ(load->indices.size(), load->buffer->shape.size())
        << "SG2260E RV tile region rank mismatch for " << view.buffer_name;
    if (global) {
      auto backing = global_addresses_.find(view.buffer_name);
      ICHECK(backing != global_addresses_.end())
          << "SG2260E RV global buffer " << view.buffer_name
          << " is not mapped to a PrimFunc parameter";
      backing_address = backing->second;
    }
    Array<PrimExpr> strides = load->buffer->strides;
    if (strides.empty()) {
      PrimExpr product = IntImm(DataType::Int(64), 1);
      std::vector<PrimExpr> reversed;
      for (int i = static_cast<int>(load->buffer->shape.size()) - 1; i >= 0; --i) {
        reversed.push_back(product);
        product = product * load->buffer->shape[i];
      }
      for (auto it = reversed.rbegin(); it != reversed.rend(); ++it)
        strides.push_back(*it);
    }
    for (size_t i = 0; i < load->indices.size(); ++i)
      element_offset = element_offset + load->indices[i] * strides[i];
  } else {
    LOG(FATAL) << "SG2260E RV tensor view " << view.buffer_name
               << " address must be tvm_access_ptr or tl.region";
  }

  if (!global) {
    auto address_it = local_addresses_.find(view.buffer_name);
    ICHECK(address_it != local_addresses_.end())
        << "SG2260E RV local buffer " << view.buffer_name
        << " has no AddressAssign function attribute";
    backing_address = std::to_string(address_it->second);
  }
  int bytes = (view.dtype_bits + 7) / 8;
  std::string offset = element_offset.as<IntImmNode>()
                           ? std::to_string(element_offset.as<IntImmNode>()->value)
                           : PrintExpr(element_offset);
  std::string address = "(" + backing_address + " + " +
                        offset + " * " + std::to_string(bytes) + ")";
  std::string payload = std::to_string(view.layout) + ":" +
                        std::to_string(view.dtype_code) + ":" +
                        std::to_string(view.dtype_bits) + ":" + address;
  for (const PrimExpr &dim : view.shape)
    payload += ":" + PrintExpr(dim);
  for (const PrimExpr &stride : view.stride)
    payload += ":" + PrintExpr(stride);
  int register_key = (view.register_class << 16) | view.register_number;
  auto configured = configured_registers_.find(register_key);
  if (configured != configured_registers_.end() && configured->second == payload)
    return;
  configured_registers_[register_key] = payload;
  PrintIndent();
  stream << (global ? "RVT_CFGGR(" : "RVT_CFGTR(") << view.register_number
         << ", " << (view.dtype_code == DataType::kInt || view.dtype_code == DataType::kUInt)
         << ", " << ElementWidth(view.dtype_code, view.dtype_bits)
         << ", 0, " << LayoutName(view.layout) << ", ";
  stream << address;
  stream << ");\n";
  PrintIndent();
  stream << (global ? "RVT_CFGGR_REG_SHAPE(" : "RVT_CFGTR_SHAPE(")
         << view.register_number;
  for (const PrimExpr &dim : view.shape)
    stream << ", " << PrintExpr(dim);
  stream << ");\n";
  if (view.layout == static_cast<int>(tl::rv::LayoutKind::kFree)) {
    PrintIndent();
    // PPL 1.7 packs the first two values as NC stride and the last two as HW.
    stream << (global ? "RVT_GR_STRIDE(" : "RVT_TR_STRIDE(")
           << view.register_number << ", FREE_LAYOUT";
    for (const PrimExpr &stride : view.stride)
      stream << ", " << PrintExpr(stride);
    stream << ");\n";
  }
}

void CodeGenTileLangPPLRV::ValidateSameTensor(
    const TensorView &lhs, const TensorView &rhs,
    const std::string &operation) const {
  ICHECK_EQ(lhs.dtype_code, rhs.dtype_code) << operation << " dtype code mismatch";
  ICHECK_EQ(lhs.dtype_bits, rhs.dtype_bits) << operation << " dtype width mismatch";
  ICHECK_EQ(lhs.dtype_lanes, rhs.dtype_lanes) << operation << " dtype lanes mismatch";
  ICHECK_EQ(lhs.shape.size(), rhs.shape.size());
  for (size_t i = 0; i < lhs.shape.size(); ++i)
    ICHECK(tvm::StructuralEqual()(lhs.shape[i], rhs.shape[i]))
        << operation << " shape mismatch";
}

void CodeGenTileLangPPLRV::VisitStmt_(const LetStmtNode *op) {
  const auto *call = op->value.as<CallNode>();
  if (call && call->op.same_as(tir::builtin::call_extern()) && !call->args.empty()) {
    const auto *name = call->args[0].as<StringImmNode>();
    if (name && name->value == tl::rv::kTensorView) {
      TensorView view = ParseTensorView(call);
      tensor_views_.emplace(op->var.get(), view);
      EmitTensorView(view);
      PrintStmt(op->body);
      tensor_views_.erase(op->var.get());
      return;
    }
  }
  CodeGenC::VisitStmt_(op);
}

const CodeGenTileLangPPLRV::TensorView &
CodeGenTileLangPPLRV::GetTensorView(const PrimExpr &expr,
                                    const std::string &operation) const {
  const auto *var = expr.as<VarNode>();
  ICHECK(var && tensor_views_.count(var))
      << operation << " expects an RV tensor_view operand";
  return tensor_views_.at(var);
}

void CodeGenTileLangPPLRV::VisitStmt_(const EvaluateNode *op) {
  const auto *call = op->value.as<CallNode>();
  if (!call || !call->op.same_as(tir::builtin::call_extern()) || call->args.empty()) {
    CodeGenC::VisitStmt_(op);
    return;
  }
  const auto *name_node = call->args[0].as<StringImmNode>();
  std::string extern_name = name_node ? std::string(name_node->value) : "";
  if (!name_node || extern_name.rfind("ppl.rv.", 0) != 0) {
    CodeGenC::VisitStmt_(op);
    return;
  }
  std::string name = extern_name;
  if (name == "ppl.rv.copy") {
    ICHECK_EQ(call->args.size(), 3U) << "ppl.rv.copy expects src and dst";
    const TensorView &src = GetTensorView(call->args[1], name);
    const TensorView &dst = GetTensorView(call->args[2], name);
    bool src_global = src.register_class == static_cast<int>(tl::rv::RegisterClass::kGlobal);
    bool dst_global = dst.register_class == static_cast<int>(tl::rv::RegisterClass::kGlobal);
    ICHECK_NE(src_global, dst_global)
        << "ppl.rv.copy currently supports global/local transfers only";
    ValidateSameTensor(src, dst, name);
    ICHECK_NE(src.dtype_code, DataType::kUInt)
        << "ppl.rv.copy does not yet support unsigned element types; "
           "unsigned RV subtype handling is deferred to the gather slice";
    ICHECK_EQ(src.access, 0) << "ppl.rv.copy src must be read-only";
    ICHECK_EQ(dst.access, 1) << "ppl.rv.copy dst must be write-only";
    PrintIndent();
    if (src_global)
      stream << "rvt_dma_ld(" << dst.register_number << ", " << src.register_number << ");\n";
    else
      stream << "rvt_dma_st(" << dst.register_number << ", " << src.register_number << ");\n";
    return;
  }
  if (name == "ppl.rv.add") {
    ICHECK_EQ(call->args.size(), 4U) << "ppl.rv.add expects dst, lhs and rhs";
    const TensorView &dst = GetTensorView(call->args[1], name);
    const TensorView &lhs = GetTensorView(call->args[2], name);
    const TensorView &rhs = GetTensorView(call->args[3], name);
    ICHECK_EQ(dst.register_class, static_cast<int>(tl::rv::RegisterClass::kTensor));
    ICHECK_EQ(lhs.register_class, static_cast<int>(tl::rv::RegisterClass::kTensor));
    ICHECK_EQ(rhs.register_class, static_cast<int>(tl::rv::RegisterClass::kTensor));
    ValidateSameTensor(dst, lhs, name);
    ValidateSameTensor(dst, rhs, name);
    ICHECK_EQ(dst.dtype_code, DataType::kFloat);
    ICHECK_EQ(dst.dtype_bits, 16) << "ppl.rv.add initially supports fp16 only";
    ICHECK_EQ(dst.access, 1);
    ICHECK_EQ(lhs.access, 0);
    ICHECK_EQ(rhs.access, 0);
    PrintIndent();
    stream << "rvt_cfg_satu(0, 0);\n";
    PrintIndent();
    stream << "rvt_cfg_round_mode(0);\n";
    PrintIndent();
    stream << "rvt_fadd(" << dst.register_number << ", " << lhs.register_number
           << ", " << rhs.register_number << ");\n";
    return;
  }
  if (name == "ppl.rv.topk") {
    LOG(FATAL) << "ppl.rv.topk is unavailable: PPL 1.7 SG2260E RV lowering "
                  "fails after converting top-k operands to register descriptors";
  }
  LOG(FATAL) << "CodeGenTileLangPPLRV does not implement " << name
             << "; refusing to fall back to atomic TPU APIs";
}

std::string BuildTileLangPPLRV(IRModule mod) {
  CodeGenTileLangPPLRV cg;
  cg.Init(false);
  for (const auto &entry : mod->functions) {
    ICHECK(entry.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangPPLRV can only take PrimFunc";
    cg.AddFunction(Downcast<PrimFunc>(entry.second));
  }
  return cg.Finish();
}

TVM_REGISTER_GLOBAL("target.build.tilelang_ppl_rv")
    .set_body_typed(BuildTileLangPPLRV);

} // namespace codegen
} // namespace tvm
