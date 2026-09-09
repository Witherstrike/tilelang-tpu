// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "codegen_ppl_rv.h"

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>

#include <cstdint>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <tvm/tir/stmt_functor.h>

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

const char *DTypeName(int code, int bits) {
  if (code == DataType::kFloat && bits == 16)
    return "DT_FP16";
  if (code == DataType::kFloat && bits == 32)
    return "DT_FP32";
  if (code == DataType::kInt && bits == 32)
    return "DT_INT32";
  if (code == DataType::kUInt && bits == 32)
    return "DT_UINT32";
  LOG(FATAL) << "Unsupported SG2260E RV scalar dtype code=" << code
             << ", bits=" << bits;
  return "";
}

} // namespace

CodeGenTileLangPPLRV::CodeGenTileLangPPLRV() { restrict_keyword_ = ""; }

std::string CodeGenTileLangPPLRV::Finish() {
  decl_stream << "#include \"tpu_kernel.h\"\n"
              << "#include \"ppl_helper.h\"\n"
              << "#include \"atomic_def.h\"\n"
              << "#include \"rvt_api.h\"\n\n";
  decl_stream << "#include <math.h>\n";
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
  scratch_tr_ = 8;
  scratch_cr_ = 1;
  tir::PostOrderVisit(f->body, [&](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (!call || !call->op.same_as(tir::builtin::call_extern()) || call->args.empty()) return;
    const auto *name = call->args[0].as<StringImmNode>();
    if (!name) return;
    if (name->value == tl::rv::kTensorView && call->args.size() == 20 &&
        AsInt(call->args[3], "class") == 1)
      scratch_tr_ = std::max(scratch_tr_, AsInt(call->args[4], "register") + 1);
    if (name->value == tl::rv::kScalar && call->args.size() == 6)
      scratch_cr_ = std::max(scratch_cr_, AsInt(call->args[1], "register") + 1);
  });
  register_addresses_.clear();
  tensor_views_.clear();
  scalar_views_.clear();
  configured_registers_.clear();
  local_addresses_.clear();
  global_addresses_.clear();
  parallel_region_ = false;
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
  // TileLang's host launcher resolves the fixed public name main_kernel.
  // Keep the symbol-derived entry for corpus/debug parity and register a thin
  // ABI wrapper, as the atomic backend does.
  stream << "int main_kernel(const void *args) {\n"
         << "  return " << function_name << "_entry(args);\n"
         << "}\n"
         << "TPUKERNEL_FUNC_REGISTER(main_kernel)\n";
}

void CodeGenTileLangPPLRV::VisitStmt_(const AllocateNode *op) {
  PrintStmt(op->body);
}

void CodeGenTileLangPPLRV::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "tpu_parallel_start") {
    ICHECK(!parallel_region_) << "Nested SG2260E RV parallel regions are unsupported";
    parallel_region_ = true;
    PrintIndent();
    stream << "tpu_parallel_start();\n";
    PrintStmt(op->body);
    return;
  }
  if (op->attr_key == "tpu_parallel_end") {
    ICHECK(parallel_region_) << "Unmatched SG2260E RV parallel region end";
    PrintIndent();
    stream << "tpu_parallel_end();\n";
    parallel_region_ = false;
    PrintStmt(op->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
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
  register_addresses_[view.register_number] = address;
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
         // PPL 1.7's SG2260E gather golden configures uint32 indices with
         // is_int=0; preserve that generated descriptor encoding exactly.
         << ", " << (view.dtype_code == DataType::kInt)
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
    if (name && name->value == tl::rv::kScalar) {
      ICHECK_EQ(call->args.size(), 6U)
          << "SG2260E RV scalar schema mismatch: expected 6 arguments";
      ScalarView scalar{AsInt(call->args[1], "scalar register"),
                        AsInt(call->args[2], "scalar dtype code"),
                        AsInt(call->args[3], "scalar dtype bits"),
                        AsInt(call->args[4], "scalar dtype lanes"),
                        call->args[5]};
      ICHECK_EQ(scalar.dtype_lanes, 1)
          << "SG2260E RV supports scalar element dtypes only";
      scalar_views_.emplace(op->var.get(), scalar);
      PrintStmt(op->body);
      scalar_views_.erase(op->var.get());
      return;
    }
  }
  CodeGenC::VisitStmt_(op);
}

const CodeGenTileLangPPLRV::ScalarView &
CodeGenTileLangPPLRV::GetScalarView(const PrimExpr &expr,
                                    const std::string &operation) const {
  const auto *var = expr.as<VarNode>();
  ICHECK(var && scalar_views_.count(var))
      << operation << " expects an RV scalar operand";
  return scalar_views_.at(var);
}

const CodeGenTileLangPPLRV::TensorView &
CodeGenTileLangPPLRV::GetTensorView(const PrimExpr &expr,
                                    const std::string &operation) const {
  const auto *var = expr.as<VarNode>();
  ICHECK(var && tensor_views_.count(var))
      << operation << " expects an RV tensor_view operand";
  return tensor_views_.at(var);
}

void CodeGenTileLangPPLRV::VisitExpr_(const FloatImmNode *op, std::ostream &os) {
  if (std::isnan(op->value)) os << "NAN";
  else if (std::isinf(op->value)) os << (op->value < 0 ? "-INFINITY" : "INFINITY");
  else {
    std::ostringstream literal;
    literal << std::scientific << std::setprecision(17) << op->value;
    if (op->dtype.bits() <= 32) literal << 'f';
    os << literal.str();
  }
}

void CodeGenTileLangPPLRV::EmitScalar(const ScalarView &value, const TensorView &dst) {
    ICHECK(value.dtype_code == DataType::kFloat ||
           value.dtype_code == DataType::kInt ||
           value.dtype_code == DataType::kUInt)
        << "ppl.rv.fill value must be numeric";
    std::string temporary = "rv_scalar_" + std::to_string(value.register_number);
    PrintIndent();
    stream << "{\n";
    PrintIndent();
    stream << "scalar_t " << temporary << " = {.u32 = 0};\n";
    PrintIndent();
    if (value.dtype_code == DataType::kFloat)
      stream << temporary << ".f32 = (float)(" << PrintExpr(value.value) << ");\n";
    else if (value.dtype_code == DataType::kUInt)
      stream << temporary << ".u32 = (uint32_t)(" << PrintExpr(value.value) << ");\n";
    else
      stream << temporary << ".s32 = (int32_t)(" << PrintExpr(value.value) << ");\n";
    PrintIndent();
    stream << temporary << " = tpu_cast(" << temporary << ", "
           << DTypeName(dst.dtype_code, dst.dtype_bits) << ", "
           << (value.dtype_code == DataType::kFloat
                   ? "DT_FP32"
                   : (value.dtype_code == DataType::kUInt ? "DT_UINT32"
                                                          : "DT_INT32"))
           << ", RM_HALF_TO_EVEN);\n";
    PrintIndent();
    stream << "RVT_CR(" << value.register_number << ", " << (dst.dtype_code == DataType::kInt) << ", "
           << ElementWidth(dst.dtype_code, dst.dtype_bits)
           << ", 0, " << temporary << ".u32);\n";
    PrintIndent();
    stream << "}\n";
}

void CodeGenTileLangPPLRV::EmitConstant(int reg, double value, const TensorView &type) {
  ICHECK_LT(reg, 8) << "RV composite operation exhausted control registers";
  EmitScalar(ScalarView{reg, DataType::kFloat, 32, 1,
      FloatImm(DataType::Float(32), value)}, type);
}

void CodeGenTileLangPPLRV::EmitLocalSlice(const TensorView &view, int reg,
    const std::string &offset, int width, int step) {
  ICHECK_EQ(view.register_class, 1) << "RV slice requires local tensor";
  ICHECK_LT(reg, 32) << "RV composite operation exhausted tensor registers";
  ICHECK_EQ(AsInt(view.shape[0], "N"), 1);
  ICHECK_EQ(AsInt(view.shape[2], "H"), 1);
  ICHECK_EQ(view.layout, 0) << "RV composite slices require HW-aligned backing tiles";
  int channels = AsInt(view.shape[1], "C");
  int bytes = view.dtype_bits / 8;
  int pitch = ((AsInt(view.shape[3], "W") * bytes + 63) / 64) * 64 / bytes;
  stream << "RVT_CFGTR(" << reg << ", " << (view.dtype_code == DataType::kInt)
         << ", " << ElementWidth(view.dtype_code, view.dtype_bits)
         << ", 0, FREE_LAYOUT, " << register_addresses_.at(view.register_number)
         << " + (" << offset << ") * " << bytes << ");\n"
         << "RVT_CFGTR_SHAPE(" << reg << ", 1, " << channels << ", 1, " << width << ");\n"
         << "RVT_TR_STRIDE(" << reg << ", FREE_LAYOUT, "
         << ((channels + 63) / 64) * pitch << ", " << pitch << ", "
         << pitch << ", " << step << ");\n";
  configured_registers_.erase((1 << 16) | reg);
}

void CodeGenTileLangPPLRV::EmitExp(const TensorView &dst, const TensorView &src,
    const TensorView &work0, const TensorView &work1) {
  for (const auto *view : {&dst, &src, &work0, &work1}) {
    ValidateSameTensor(dst, *view, "ppl.rv.exp");
    ICHECK_EQ(view->register_class, 1);
    ICHECK_EQ(view->dtype_code, DataType::kFloat);
    ICHECK_EQ(view->dtype_bits, 32) << "exp requires FP32 local tensors";
  }
  ICHECK_NE(work0.buffer_name, work1.buffer_name);
  ICHECK_NE(dst.buffer_name, work0.buffer_name);
  ICHECK_NE(dst.buffer_name, work1.buffer_name);
  int d = dst.register_number, x = src.register_number;
  int n = work0.register_number, r = work1.register_number, c = scratch_cr_;
  stream << "rvt_cfg_satu(0, 0);\nrvt_cfg_round_mode(0);\n"
         << "rvt_cp(" << r << ", " << x << ");\n";
  // exp(x) = exp(x/2)^2 keeps the exponent reconstruction in the normal
  // FP32 range even when the final result is subnormal or overflows.
  EmitConstant(c, -104, dst);
  stream << "rvt_fmax(" << d << ", " << x << ", " << c << ");\n";
  EmitConstant(c, 89, dst);
  stream << "rvt_fmin(" << d << ", " << d << ", " << c << ");\n";
  // Restore NaNs after clamping, including in-place exp.
  stream << "rvt_fcmpeq(" << d << ", " << r << ", " << r << ", " << d << ", " << r << ");\n";
  EmitConstant(c, 0.5, dst);
  stream << "rvt_fmul(" << d << ", " << d << ", " << c << ");\n";
  EmitConstant(c, 1.4426950408889634, dst);
  stream << "rvt_fmul(" << r << ", " << d << ", " << c << ");\n"
         << "rvt_cfg_round_mode(0);\n";
  TensorView integer = work0;
  integer.dtype_code = DataType::kInt;
  EmitTensorView(integer);
  stream << "rvt_cvt_f2i(" << n << ", " << r << ");\n"
         << "rvt_cvt_i2f(" << r << ", " << n << ");\n";
  EmitConstant(c, 0.6931471805599453, dst);
  stream << "rvt_fmul(" << r << ", " << r << ", " << c << ");\n"
         << "rvt_fsub(" << d << ", " << d << ", " << r << ");\n";
  // Horner polynomial on [-ln(2)/2, ln(2)/2].
  const double coefficients[] = {1, 1, 0.5, 1.0/6, 1.0/24, 1.0/120, 1.0/720, 1.0/5040};
  EmitConstant(c, coefficients[7], dst);
  stream << "rvt_cp(" << r << ", " << c << ");\n";
  for (int i = 6; i >= 0; --i) {
    stream << "rvt_fmul(" << r << ", " << r << ", " << d << ");\n";
    EmitConstant(c, coefficients[i], dst);
    stream << "rvt_fadd(" << r << ", " << r << ", " << c << ");\n";
  }
  EmitConstant(c, 127, integer);
  stream << "rvt_add(" << n << ", " << n << ", " << c << ", 0, 0);\n";
  EmitConstant(c, 8388608, integer);
  stream << "rvt_mul(" << n << ", " << n << ", " << c << ", 0, 0);\n";
  EmitTensorView(work0);
  stream << "rvt_fmul(" << d << ", " << r << ", " << n << ");\n"
         << "rvt_fmul(" << d << ", " << d << ", " << d << ");\n";
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
  if (name == "ppl.rv.exp" || name == "ppl.rv.sigmoid") {
    bool sigmoid = name == "ppl.rv.sigmoid";
    ICHECK_EQ(call->args.size(), sigmoid ? 7U : 6U);
    const auto &dst = GetTensorView(call->args[1], name);
    const auto &src = sigmoid ? GetTensorView(call->args[2], name) : dst;
    const auto &work0 = GetTensorView(call->args[sigmoid ? 3 : 2], name);
    const auto &work1 = GetTensorView(call->args[sigmoid ? 4 : 3], name);
    if (sigmoid) {
      EmitConstant(scratch_cr_, -1, dst);
      stream << "rvt_fmul(" << dst.register_number << ", " << src.register_number << ", " << scratch_cr_ << ");\n";
    }
    EmitExp(dst, sigmoid ? dst : src, work0, work1);
    if (sigmoid) {
      EmitConstant(scratch_cr_, 1, dst);
      stream << "rvt_fadd(" << dst.register_number << ", " << dst.register_number << ", " << scratch_cr_ << ");\n"
             << "rvt_fdiv(" << dst.register_number << ", " << scratch_cr_ << ", " << dst.register_number << ");\n";
    }
    return;
  }
  if (name == "ppl.rv.reduce_sum" || name == "ppl.rv.reduce_max") {
    ICHECK_EQ(call->args.size(), 7U) << name << " expects src, dst, tmp and three geometry operands";
    const auto &src = GetTensorView(call->args[1], name);
    const auto &dst = GetTensorView(call->args[2], name);
    ICHECK_EQ(src.dtype_code, DataType::kFloat);
    ICHECK_EQ(dst.dtype_code, src.dtype_code);
    ICHECK_EQ(dst.dtype_bits, src.dtype_bits);
    ICHECK_EQ(AsInt(dst.shape[3], "reduce width"), 1);
    ICHECK_EQ(AsInt(dst.shape[1], "reduce channels"), AsInt(src.shape[1], "channels"));
    ICHECK_EQ(dst.register_class, 1);
    ICHECK_NE(src.buffer_name, dst.buffer_name) << name << " requires separate input and output";
    stream << "rvt_cfg_satu(0, 0);\nrvt_cfg_round_mode(0);\n";
    if (name == "ppl.rv.reduce_sum") {
      EmitConstant(scratch_cr_, 0, dst);
      stream << "rvt_cp(" << dst.register_number << ", " << scratch_cr_ << ");\n";
    }
    stream << "{\nfor (int rv_column = 0; rv_column < " << AsInt(src.shape[3], "width") << "; ++rv_column) {\n";
    EmitLocalSlice(src, scratch_tr_, "rv_column", 1);
    stream << (name == "ppl.rv.reduce_sum" ? "rvt_fadd(" : "rvt_fmax(")
           << dst.register_number << ", " << dst.register_number << ", " << scratch_tr_ << ");\n}\n}\n";
    return;
  }
  if (name == "ppl.rv.rope_add") {
    ICHECK_EQ(call->args.size(), 6U);
    const auto &dst = GetTensorView(call->args[1], name);
    ICHECK_EQ(dst.dtype_code, DataType::kFloat);
    int width = AsInt(dst.shape[3], "rope width");
    ICHECK_EQ(width % 2, 0);
    stream << "rvt_cfg_satu(0, 0);\nrvt_cfg_round_mode(0);\n";
    for (int parity = 0; parity < 2; ++parity) {
      const auto &a = GetTensorView(call->args[parity ? 4 : 2], name);
      const auto &b = GetTensorView(call->args[parity ? 5 : 3], name);
      ValidateSameTensor(dst, a, name);
      ValidateSameTensor(dst, b, name);
      ICHECK_NE(dst.buffer_name, a.buffer_name) << "rope_add requires separate output";
      ICHECK_NE(dst.buffer_name, b.buffer_name) << "rope_add requires separate output";
      EmitLocalSlice(dst, scratch_tr_, std::to_string(parity), width / 2, 2);
      EmitLocalSlice(a, scratch_tr_ + 1, std::to_string(parity), width / 2, 2);
      EmitLocalSlice(b, scratch_tr_ + 2, std::to_string(1 - parity), width / 2, 2);
      stream << "rvt_fadd(" << scratch_tr_ << ", " << scratch_tr_ + 1 << ", " << scratch_tr_ + 2 << ");\n";
    }
    return;
  }
  if (name == "ppl.rv.copy") {
    ICHECK_EQ(call->args.size(), 3U) << "ppl.rv.copy expects src and dst";
    const TensorView &src = GetTensorView(call->args[1], name);
    const TensorView &dst = GetTensorView(call->args[2], name);
    bool src_global = src.register_class == static_cast<int>(tl::rv::RegisterClass::kGlobal);
    bool dst_global = dst.register_class == static_cast<int>(tl::rv::RegisterClass::kGlobal);
    for (size_t i = 0; i < src.shape.size(); ++i)
      ICHECK(tvm::StructuralEqual()(src.shape[i], dst.shape[i])) << name << " shape mismatch";
    bool same_dtype = src.dtype_code == dst.dtype_code && src.dtype_bits == dst.dtype_bits;
    if (src_global || dst_global)
      ICHECK(same_dtype) << "RV DMA copy requires matching dtypes; cast in local memory first";
    ICHECK_EQ(src.access, 0) << "ppl.rv.copy src must be read-only";
    ICHECK_EQ(dst.access, 1) << "ppl.rv.copy dst must be write-only";
    PrintIndent();
    if (src_global && dst_global)
      stream << "rvt_dma_scp(" << dst.register_number << ", " << src.register_number << ");\n";
    else if (!src_global && !dst_global) {
      std::string instruction = "rvt_cp";
      if (!same_dtype) {
        instruction = "rvt_cvt_";
        instruction += src.dtype_code == DataType::kFloat ? "f2" : "i2";
        instruction += dst.dtype_code == DataType::kFloat ? "f" : "i";
      }
      stream << "rvt_cfg_round_mode(0);\n" << instruction << "(" << dst.register_number << ", " << src.register_number << ");\n";
    } else if (src_global)
      stream << "rvt_dma_ld(" << dst.register_number << ", " << src.register_number << ");\n";
    else
      stream << "rvt_dma_st(" << dst.register_number << ", " << src.register_number << ");\n";
    return;
  }
  if (name == "ppl.rv.add" || name == "ppl.rv.sub" ||
      name == "ppl.rv.mul" || name == "ppl.rv.div") {
    ICHECK_EQ(call->args.size(), 4U) << "ppl.rv.add expects dst, lhs and rhs";
    const TensorView &dst = GetTensorView(call->args[1], name);
    const TensorView &lhs = GetTensorView(call->args[2], name);
    const TensorView &rhs = GetTensorView(call->args[3], name);
    ICHECK_EQ(dst.register_class, static_cast<int>(tl::rv::RegisterClass::kTensor));
    ICHECK_EQ(lhs.register_class, static_cast<int>(tl::rv::RegisterClass::kTensor));
    ICHECK_EQ(rhs.register_class, static_cast<int>(tl::rv::RegisterClass::kTensor));
    ValidateSameTensor(dst, lhs, name);
    ICHECK_EQ(dst.dtype_code, rhs.dtype_code);
    ICHECK_EQ(dst.dtype_bits, rhs.dtype_bits);
    for (size_t i = 0; i < dst.shape.size(); ++i)
      ICHECK(AsInt(rhs.shape[i], "broadcast extent") == 1 ||
             tvm::StructuralEqual()(dst.shape[i], rhs.shape[i])) << name << " incompatible broadcast";
    ICHECK_EQ(dst.dtype_code, DataType::kFloat);
    ICHECK(dst.dtype_bits == 16 || dst.dtype_bits == 32)
        << name << " supports fp16/fp32 only";
    ICHECK_EQ(dst.access, 1);
    ICHECK_EQ(lhs.access, 0);
    ICHECK_EQ(rhs.access, 0);
    PrintIndent();
    stream << "rvt_cfg_satu(0, 0);\n";
    PrintIndent();
    stream << "rvt_cfg_round_mode(0);\n";
    PrintIndent();
    stream << "rvt_f" << name.substr(7) << "(" << dst.register_number << ", " << lhs.register_number
           << ", " << rhs.register_number << ");\n";
    return;
  }
  if (name == "ppl.rv.add_C" || name == "ppl.rv.mul_C") {
    ICHECK_EQ(call->args.size(), 4U);
    const TensorView &dst = GetTensorView(call->args[1], name);
    const TensorView &src = GetTensorView(call->args[2], name);
    const ScalarView &value = GetScalarView(call->args[3], name);
    ValidateSameTensor(dst, src, name);
    ICHECK_EQ(dst.register_class, static_cast<int>(tl::rv::RegisterClass::kTensor));
    ICHECK_EQ(src.register_class, static_cast<int>(tl::rv::RegisterClass::kTensor));
    ICHECK_EQ(dst.dtype_code, DataType::kFloat);
    ICHECK(dst.dtype_bits == 16 || dst.dtype_bits == 32);
    EmitScalar(value, dst);
    PrintIndent();
    stream << "rvt_cfg_satu(0, 0);\nrvt_cfg_round_mode(0);\n";
    PrintIndent();
    stream << (name == "ppl.rv.add_C" ? "rvt_fadd(" : "rvt_fmul(")
           << dst.register_number << ", " << src.register_number << ", "
           << value.register_number << ");\n";
    return;
  }
  if (name == "ppl.rv.fill") {
    ICHECK_EQ(call->args.size(), 3U) << "ppl.rv.fill expects dst and value";
    const TensorView &dst = GetTensorView(call->args[1], name);
    const ScalarView &value = GetScalarView(call->args[2], name);
    ICHECK_EQ(dst.register_class, static_cast<int>(tl::rv::RegisterClass::kTensor));
    ICHECK_EQ(dst.access, 1) << "ppl.rv.fill dst must be write-only";
    ICHECK_EQ(dst.dtype_code, DataType::kFloat);
    ICHECK(dst.dtype_bits == 16 || dst.dtype_bits == 32)
        << "ppl.rv.fill supports fp16/fp32 destinations only";
    EmitScalar(value, dst);
    PrintIndent();
    stream << "rvt_cp(" << dst.register_number << ", "
           << value.register_number << ");\n";
    return;
  }
  if (name == "ppl.rv.gemm") {
    ICHECK_EQ(call->args.size(), 9U)
        << "ppl.rv.gemm expects lhs, rhs, accumulator, transpose flags and M/N/K";
    const TensorView &lhs = GetTensorView(call->args[1], name);
    const TensorView &rhs = GetTensorView(call->args[2], name);
    const TensorView &dst = GetTensorView(call->args[3], name);
    for (const TensorView *view : {&lhs, &rhs}) {
      ICHECK_EQ(view->register_class,
                static_cast<int>(tl::rv::RegisterClass::kTensor));
      ICHECK_EQ(view->dtype_code, DataType::kFloat);
      ICHECK_EQ(view->dtype_bits, 16)
          << "ppl.rv.gemm fmm2a inputs must be fp16";
    }
    ICHECK_EQ(dst.register_class,
              static_cast<int>(tl::rv::RegisterClass::kTensor));
    ICHECK_EQ(dst.dtype_code, DataType::kFloat);
    ICHECK_EQ(dst.dtype_bits, 32)
        << "ppl.rv.gemm rvt_fmm2a accumulator must be fp32; PPL 1.7 does "
           "not define fp16 accumulation";
    ICHECK_EQ(lhs.access, 0);
    ICHECK_EQ(rhs.access, 0);
    ICHECK_EQ(dst.access, 2) << "ppl.rv.gemm output must be a read-write accumulator";
    bool transpose_lhs = false;
    bool transpose_rhs = false;
    if (call->args.size() > 4) {
      const auto *transpose = call->args[4].as<IntImmNode>();
      ICHECK(transpose) << "ppl.rv.gemm lhs transpose flag must be constant";
      transpose_lhs = transpose->value != 0;
    }
    if (call->args.size() > 5) {
      const auto *transpose = call->args[5].as<IntImmNode>();
      ICHECK(transpose) << "ppl.rv.gemm rhs transpose flag must be constant";
      transpose_rhs = transpose->value != 0;
    }
    ICHECK(!transpose_lhs || transpose_rhs)
        << "ppl.rv.gemm transpose_A=True, transpose_B=False has no SG2260E "
           "RV fmm2 variant";
    int m = AsInt(call->args[6], "gemm M");
    int n = AsInt(call->args[7], "gemm N");
    int k = AsInt(call->args[8], "gemm K");
    ICHECK_GT(m, 0);
    ICHECK_GT(n, 0);
    ICHECK_GT(k, 0);
    ICHECK_EQ(AsInt(dst.shape[1], "gemm accumulator C"), m);
    ICHECK_EQ(AsInt(dst.shape[3], "gemm accumulator W"), n);
    ICHECK_EQ(AsInt(lhs.shape[1], "gemm lhs C"), transpose_lhs ? k : m);
    ICHECK_EQ(AsInt(lhs.shape[3], "gemm lhs W"), transpose_lhs ? m : k);
    ICHECK_EQ(AsInt(rhs.shape[1], "gemm rhs C"), transpose_rhs ? n : k);
    ICHECK_EQ(AsInt(rhs.shape[3], "gemm rhs W"), transpose_rhs ? k : n);
    PrintIndent();
    stream << "rvt_cfg_quant(0);\n";
    PrintIndent();
    stream << "rvt_cfg_satu(0, 0);\n";
    PrintIndent();
    stream << (transpose_lhs ? "rvt_fmm2a_tt(" :
               transpose_rhs ? "rvt_fmm2a_nt(" : "rvt_fmm2a_nn(")
           << dst.register_number << ", "
           << lhs.register_number << ", " << rhs.register_number
           << ", 0, 0, 0);\n";
    return;
  }
  if (name == "ppl.rv.rsqrt") {
    ICHECK_GE(call->args.size(), 3U) << "ppl.rv.rsqrt expects dst and src";
    const TensorView &dst = GetTensorView(call->args[1], name);
    const TensorView &src = GetTensorView(call->args[2], name);
    ValidateSameTensor(dst, src, name);
    ICHECK_EQ(dst.register_class, static_cast<int>(tl::rv::RegisterClass::kTensor));
    ICHECK_EQ(src.register_class, static_cast<int>(tl::rv::RegisterClass::kTensor));
    ICHECK_EQ(dst.dtype_code, DataType::kFloat);
    ICHECK(dst.dtype_bits == 16 || dst.dtype_bits == 32)
        << "ppl.rv.rsqrt supports FP16/FP32";
    int iterations = 3;
    if (call->args.size() > 3) {
      const auto *iter = call->args[3].as<IntImmNode>();
      ICHECK(iter) << "ppl.rv.rsqrt iteration count must be constant";
      iterations = static_cast<int>(iter->value);
    }
    ICHECK_GT(iterations, 0); ICHECK_LE(iterations, 8);
    PrintIndent();
    stream << "rvt_cfg_rsqrt_iter(" << iterations << ");\n";
    PrintIndent();
    stream << "rvt_sfu_rsqrt(" << dst.register_number << ", "
           << src.register_number << ");\n";
    return;
  }
  if (name == "ppl.rv.gather") {
    ICHECK_GE(call->args.size(), 4U)
        << "ppl.rv.gather expects dst, parameter and index";
    const TensorView &dst = GetTensorView(call->args[1], name);
    const TensorView &param = GetTensorView(call->args[2], name);
    const TensorView &index = GetTensorView(call->args[3], name);
    for (const TensorView *view : {&dst, &param, &index})
      ICHECK_EQ(view->register_class,
                static_cast<int>(tl::rv::RegisterClass::kGlobal))
          << "ppl.rv.gather golden path requires global tensors";
    ICHECK_EQ(dst.access, 1);
    ICHECK_EQ(param.access, 0);
    ICHECK_EQ(index.access, 0);
    ICHECK_EQ(dst.dtype_code, DataType::kFloat);
    ICHECK_EQ(dst.dtype_bits, 16);
    ICHECK_EQ(param.dtype_code, dst.dtype_code);
    ICHECK_EQ(param.dtype_bits, dst.dtype_bits);
    ICHECK_EQ(index.dtype_code, DataType::kUInt);
    ICHECK_EQ(index.dtype_bits, 32)
        << "ppl.rv.gather index must be uint32";
    ICHECK_EQ(call->args.size(), 5U)
        << "ppl.rv.gather expects the TileLang param_h operand";
    int param_h = AsInt(call->args[4], "gather param_h");
    ICHECK_GT(param_h, 0) << "ppl.rv.gather param_h must be positive";
    ICHECK_LE(param_h, 65535)
        << "ppl.rv.gather param_h exceeds the SG2260E GR shape field";
    int descriptor_h = AsInt(param.shape[2], "gather parameter H");
    ICHECK_EQ(param_h, descriptor_h)
        << "ppl.rv.gather param_h must match the source table's logical height";
    PrintIndent();
    stream << "{\nuint64_t rv_dma_index = 0;\n";
    PrintIndent();
    // TileLang's param_h is represented by the configured source descriptor.
    // Its API exposes neither PPL's constant fill value nor index_start_pos;
    // both are zero, matching the SG2260E gather golden.
    stream << "rvt_cfg_dmaidx(0, &rv_dma_index);\n";
    PrintIndent();
    stream << "rvt_dma_hgather(" << dst.register_number << ", "
           << param.register_number << ", " << index.register_number
           << ", 0);\n}\n";
    return;
  }
  if (name == "ppl.rv.topk") {
    ICHECK_EQ(call->args.size(), 7U) << "ppl.rv.topk expects values, indices, src, K, descending, length";
    const auto &values = GetTensorView(call->args[1], name);
    const auto &indices = GetTensorView(call->args[2], name);
    const auto &src = GetTensorView(call->args[3], name);
    int k = AsInt(call->args[4], "topk K");
    bool descending = AsInt(call->args[5], "topk descending") != 0;
    int length = AsInt(call->args[6], "topk length");
    ICHECK_GT(k, 0); ICHECK_GE(length, k);
    ICHECK_EQ(src.dtype_bits, 32); ICHECK_EQ(values.dtype_bits, 32);
    ICHECK_EQ(src.dtype_code, values.dtype_code);
    ICHECK(src.dtype_code == DataType::kFloat || src.dtype_code == DataType::kInt || src.dtype_code == DataType::kUInt);
    ICHECK_EQ(indices.dtype_bits, 32);
    ICHECK(indices.dtype_code == DataType::kInt || indices.dtype_code == DataType::kUInt);
    for (const auto *view : {&src, &values, &indices}) {
      ICHECK_EQ(view->register_class, 2) << "ppl.topk requires global buffers";
      ICHECK_EQ(view->layout, 1) << "ppl.topk requires contiguous buffers";
      int64_t elements = 1;
      for (const auto &dim : view->shape) elements *= AsInt(dim, "topk shape");
      ICHECK_GE(elements, view == &src ? length : k);
    }
    ICHECK_NE(src.buffer_name, values.buffer_name);
    ICHECK_NE(src.buffer_name, indices.buffer_name);
    ICHECK_NE(values.buffer_name, indices.buffer_name);
    auto scratch = local_addresses_.find("tir.tpu.rv.topk_scratch");
    ICHECK(scratch != local_addresses_.end()) << "Run AddressAssign with device_mode=rv before topk legalization";
    ICHECK_LT(scratch_tr_ + 6, 32); ICHECK_LT(scratch_cr_ + 3, 8);
    int a = scratch_tr_, b = a + 1, bi = a + 2, eligible = a + 3;
    int ti = a + 4, tv = a + 5, previous = a + 6;
    int zero = scratch_cr_, minus_one = zero + 1, jreg = zero + 2, one = zero + 3;
    TensorView inttype = src; inttype.dtype_code = DataType::kInt;
    EmitConstant(zero, 0, inttype); EmitConstant(minus_one, -1, inttype); EmitConstant(one, 1, inttype);
    stream << "{\n";
    for (int i = 0; i < 7; ++i) {
      bool value = i == 0 || i == 1 || i == 5;
      stream << "RVT_CFGTR(" << a+i << ", " << (value ? src.dtype_code == DataType::kInt : true)
             << ", TEEW_E32, 0, HW_ALIGN_LAYOUT, " << scratch->second + i*64 << ");\n"
             << "RVT_CFGTR_SHAPE(" << a+i << ", 1, 1, 1, 1);\n";
    }
    auto global_scalar = [&](const TensorView &v, const std::string &offset) {
      stream << "RVT_CFGGR(" << v.register_number << ", " << (v.dtype_code == DataType::kInt)
             << ", TEEW_E32, 0, CONTINUOUS_LAYOUT, " << register_addresses_.at(v.register_number)
             << " + (" << offset << ") * 4);\nRVT_CFGGR_REG_SHAPE(" << v.register_number << ", 1, 1, 1, 1);\n";
      configured_registers_.erase((2 << 16) | v.register_number);
    };
    // Stable repeated selection. Outputs from earlier ranks are used only
    // for exclusion; the input is never modified. Scratch is allocator-owned.
    stream << "for (int rv_rank=0; rv_rank<" << k << "; ++rv_rank) {\n"
           << "rvt_cp(" << bi << ", " << minus_one << ");\n"
           << "rvt_cp(" << b << ", " << zero << ");\n"
           << "for (int rv_j=0; rv_j<" << length << "; ++rv_j) {\n"
           << "RVT_CR(" << jreg << ", 1, TEEW_E32, 0, rv_j);\n";
    global_scalar(src, "rv_j");
    stream << "rvt_dma_ld(" << a << ", " << src.register_number << ");\n"
           << "rvt_cp(" << eligible << ", " << one << ");\n"
           << "for (int rv_h=0; rv_h<rv_rank; ++rv_h) {\n";
    global_scalar(indices, "rv_h");
    stream << "rvt_dma_ld(" << previous << ", " << indices.register_number << ");\n"
           << "rvt_cmpeq(" << eligible << ", " << jreg << ", " << previous << ", " << zero << ", " << eligible << ");\n}\n";
    std::string cmp = src.dtype_code == DataType::kFloat ? "rvt_fcmp" : "rvt_cmp";
    cmp += descending ? "gt" : "lt";
    stream << cmp << "(" << ti << ", " << a << ", " << b << ", " << jreg << ", " << bi << ");\n"
           << cmp << "(" << tv << ", " << a << ", " << b << ", " << a << ", " << b << ");\n";
    if (src.dtype_code == DataType::kFloat) {
      // Finite values precede NaNs in either direction; ties keep first index.
      stream << "rvt_fcmpeq(" << ti << ", " << b << ", " << b << ", " << ti << ", " << jreg << ");\n"
             << "rvt_fcmpeq(" << tv << ", " << b << ", " << b << ", " << tv << ", " << a << ");\n"
             << "rvt_fcmpeq(" << ti << ", " << a << ", " << a << ", " << ti << ", " << bi << ");\n"
             << "rvt_fcmpeq(" << tv << ", " << a << ", " << a << ", " << tv << ", " << b << ");\n";
    }
    stream << "rvt_cmpeq(" << ti << ", " << bi << ", " << minus_one << ", " << jreg << ", " << ti << ");\n"
           << "rvt_cmpeq(" << tv << ", " << bi << ", " << minus_one << ", " << a << ", " << tv << ");\n"
           << "rvt_cmpeq(" << bi << ", " << eligible << ", " << one << ", " << ti << ", " << bi << ");\n"
           << "rvt_cmpeq(" << b << ", " << eligible << ", " << one << ", " << tv << ", " << b << ");\n"
           << "rvt_sync_all();\n}\n";
    global_scalar(values, "rv_rank"); global_scalar(indices, "rv_rank");
    stream << "rvt_dma_st(" << values.register_number << ", " << b << ");\n"
           << "rvt_dma_st(" << indices.register_number << ", " << bi << ");\n"
           << "rvt_sync_all();\n}\n}\n";
    return;
  }
  LOG(FATAL) << "CodeGenTileLangPPLRV does not implement " << name
             << "; refusing to fall back to atomic TPU APIs";
}

std::string BuildTileLangPPLRV(IRModule mod) {
  ICHECK_EQ(mod->functions.size(), 1U)
      << "CodeGenTileLangPPLRV supports exactly one PrimFunc because the TPU "
         "runtime ABI exports the fixed main_kernel symbol";
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
