/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file distributed
 * with this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0.
 */

/*! \file target/codegen_rv.cc
 *  \brief SG2260E RV Tensor instruction selection for portable TPU ops.
 */

#include "codegen_tpu_common.h"

#include <tvm/runtime/logging.h>

#include <cmath>
#include <string>
#include <vector>

namespace tvm {
namespace codegen {

namespace {

std::string RVDTypeName(DataType dtype) {
  if (dtype == DataType::Float(32)) {
    return "DT_FP32";
  }
  if (dtype == DataType::Float(16)) {
    return "DT_FP16";
  }
  if (dtype == DataType::BFloat(16)) {
    return "DT_BFP16";
  }
  LOG(FATAL) << "RV Tensor floating-point lowering does not support dtype "
             << dtype;
  return "DT_FP32";
}

void CheckRVShape(const std::vector<int> &shape, const char *operation,
                  const char *operand) {
  ICHECK_EQ(shape.size(), 2U)
      << "RV Tensor " << operation << " expects a TileLang 2-D local tile for "
      << operand;
  for (int extent : shape) {
    ICHECK_GT(extent, 0) << "RV Tensor " << operation << " requires positive "
                         << operand << " extents";
    ICHECK_LT(extent, 1 << 16)
        << "RV Tensor " << operation << " descriptor extent exceeds 16 bits";
  }
}

} // namespace

void CodeGenTileLangTPU::EmitRVDescriptor(const std::string &tensor,
                                          int register_id, bool is_global,
                                          const std::string &dtype,
                                          bool hw_aligned) {
  const bool is_float = dtype == "DT_FP32" || dtype == "DT_FP16" ||
                        dtype == "DT_BFP16" || dtype == "DT_FP8E5M2" ||
                        dtype == "DT_FP8E4M3";
  PrintIndent();
  stream << (is_global ? "rvt_gr(" : "rvt_tr(") << register_id << ", PRECISION("
         << dtype << "), " << (is_float ? "FP8TYPE(" : "SIGN(") << dtype
         << "), " << tensor << ".addr, "
         << (hw_aligned ? "HW_ALIGN_LAYOUT" : "FREE_LAYOUT")
         << ", (array4_t){.n=" << tensor << ".shape.n, .c=" << tensor
         << ".shape.c, .h=" << tensor << ".shape.h, .w=" << tensor
         << ".shape.w}, "
         << (hw_aligned ? "(int *)NULL" : "(int *)&" + tensor + ".stride")
         << ");\n";
}

void CodeGenTileLangTPU::EmitRVCopy(const std::string &src, bool src_is_global,
                                    const std::string &src_dtype,
                                    const std::string &dst, bool dst_is_global,
                                    const std::string &dst_dtype) {
  // The DMA descriptors deliberately use FREE layout.  A TileLang region can
  // be a sub-view whose logical width is smaller than its parent's physical
  // HW-aligned row, so recomputing an aligned stride from the region shape is
  // not generally correct.
  const int src_register = src_is_global ? 32 : 8;
  // A single-global-operand transfer can reuse GR32.  GR33 is only needed
  // when both endpoints are global and GR32 is already occupied by src.
  const int dst_register = dst_is_global ? (src_is_global ? 33 : 32) : 9;
  EmitRVDescriptor(src, src_register, src_is_global, src_dtype, false);
  EmitRVDescriptor(dst, dst_register, dst_is_global, dst_dtype, false);

  if (src_dtype != dst_dtype) {
    ICHECK(!src_is_global && !dst_is_global)
        << "RV Tensor copy-and-convert requires two local tensors; DMA does "
           "not perform dtype conversion";
    auto is_supported_float = [](const std::string &dtype) {
      return dtype == "DT_FP16" || dtype == "DT_BFP16" || dtype == "DT_FP32";
    };
    ICHECK(is_supported_float(src_dtype) && is_supported_float(dst_dtype))
        << "RV Tensor rvt_cvt_f2f only accepts supported floating-point "
           "source/destination types; integer conversion requires a distinct "
           "rvt_cvt_i2i/i2f/f2i lowering";
    PrintIndent();
    stream << "rvt_cfg_satu(0, true);\n";
    PrintIndent();
    stream << "rvt_cfg_round_mode(0);\n";
    PrintIndent();
    stream << "rvt_cvt_f2f(" << dst_register << ", " << src_register << ");\n";
    return;
  }

  PrintIndent();
  if (src_is_global && !dst_is_global) {
    stream << "rvt_dma_ld(" << dst_register << ", " << src_register << ");\n";
  } else if (!src_is_global && dst_is_global) {
    stream << "rvt_dma_st(" << dst_register << ", " << src_register << ");\n";
  } else {
    stream << "rvt_dma_cp(" << dst_register << ", " << src_register << ");\n";
  }
}

void CodeGenTileLangTPU::EmitRVFill(const std::string &dst, DataType dtype,
                                    double value) {
  RVDTypeName(dtype); // Validate before emitting a partially formed kernel.
  ICHECK_EQ(value, 0.0)
      << "RV Tensor tl.tpu.fill currently supports the zero constant used to "
         "initialize accumulators; non-zero scalar materialization is not yet "
         "part of the portable TPU contract";
  EmitRVDescriptor(dst, 10, false, RVDTypeName(dtype), true);
  PrintIndent();
  stream << "{\n";
  PrintIndent();
  stream << "uint64_t " << dst << "_zero_bits = 0;\n";
  PrintIndent();
  stream << "rvt_cr(1, PRECISION(" << RVDTypeName(dtype) << "), FP8TYPE("
         << RVDTypeName(dtype) << "), &" << dst << "_zero_bits);\n";
  PrintIndent();
  stream << "rvt_cp(10, 1);\n";
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangTPU::EmitRVGemm(const std::string &a, const std::string &b,
                                    const std::string &c, DataType a_dtype,
                                    DataType b_dtype, DataType c_dtype,
                                    bool transpose_a, bool transpose_b,
                                    bool accumulate, int64_t m, int64_t n,
                                    int64_t k) {
  ICHECK(!transpose_a)
      << "RV Tensor high-performance GEMM does not expose a standalone TN "
         "form; transpose_A=true is not supported by tl.tpu.gemm";
  ICHECK(a_dtype == b_dtype)
      << "RV Tensor GEMM requires matching A/B dtypes, got " << a_dtype
      << " and " << b_dtype;
  ICHECK(a_dtype == DataType::Float(16) || a_dtype == DataType::BFloat(16))
      << "RV Tensor fmm2 currently accepts FP16 or BF16 TileLang inputs, got "
      << a_dtype;
  ICHECK(
      (accumulate && c_dtype == DataType::Float(32)) ||
      (!accumulate && (c_dtype == DataType::Float(32) || c_dtype == a_dtype)))
      << "RV Tensor accumulating fmm2 requires an FP32 C tile; overwrite mode "
         "permits FP32 or a C tile matching A/B, got "
      << c_dtype;
  ICHECK_GT(m, 0);
  ICHECK_GT(n, 0);
  ICHECK_GT(k, 0);
  ICHECK_LT(m, 1 << 16);
  ICHECK_LT(n, 1 << 16);
  ICHECK_LT(k, 1 << 16);

  EmitRVDescriptor(a, 8, false, RVDTypeName(a_dtype), true);
  EmitRVDescriptor(b, 9, false, RVDTypeName(b_dtype), true);
  EmitRVDescriptor(c, 10, false, RVDTypeName(c_dtype), true);
  PrintIndent();
  stream << "rvt_cfg_quant(0);\n";
  PrintIndent();
  stream << "rvt_cfg_satu(0, false);\n";
  PrintIndent();
  // Accumulation is an explicit semantic bit; the RV ISA has both variants.
  const char *instruction = nullptr;
  if (transpose_b) {
    instruction = accumulate ? "rvt_fmm2a_nt" : "rvt_fmm2_nt";
  } else {
    instruction = accumulate ? "rvt_fmm2a_nn" : "rvt_fmm2_nn";
  }
  stream << instruction << "(10, 8, 9, 0, 0, 0);\n";
}

void CodeGenTileLangTPU::EmitRVElementwise(
    const std::string &operation, const std::string &dst,
    const std::string &src0, const std::string &src1, DataType dst_dtype,
    DataType src0_dtype, DataType src1_dtype, const std::vector<int> &dst_shape,
    const std::vector<int> &src0_shape, const std::vector<int> &src1_shape) {
  CheckRVShape(dst_shape, operation.c_str(), "output");
  CheckRVShape(src0_shape, operation.c_str(), "lhs");
  CheckRVShape(src1_shape, operation.c_str(), "rhs");
  ICHECK(dst_dtype == src0_dtype && dst_dtype == src1_dtype)
      << "RV Tensor " << operation
      << " currently requires matching input/output dtypes";
  RVDTypeName(dst_dtype);
  if (operation == "div") {
    ICHECK(dst_dtype == DataType::Float(16) ||
           dst_dtype == DataType::BFloat(16) ||
           dst_dtype == DataType::Float(32))
        << "rvt_fdiv only supports FP16, BF16, and FP32";
  }
  ICHECK(dst_shape == src0_shape) << "RV Tensor " << operation
                                  << " requires output and lhs shapes to match";
  ICHECK(src1_shape == dst_shape ||
         (src1_shape[0] == dst_shape[0] && src1_shape[1] == 1))
      << "RV Tensor " << operation
      << " only supports an equal rhs shape or W-dimension broadcast for the "
         "current 2-D TileLang mapping";

  EmitRVDescriptor(src0, 8, false, RVDTypeName(src0_dtype), true);
  EmitRVDescriptor(src1, 9, false, RVDTypeName(src1_dtype), true);
  EmitRVDescriptor(dst, 10, false, RVDTypeName(dst_dtype), true);
  PrintIndent();
  stream << "rvt_cfg_satu(0, false);\n";
  PrintIndent();
  stream << "rvt_cfg_round_mode(0);\n";
  if (operation == "div") {
    PrintIndent();
    stream << "rvt_cfg_rsqrt_iter(3);\n";
  }
  PrintIndent();
  stream << "rvt_f" << operation << "(10, 8, 9);\n";
}

} // namespace codegen
} // namespace tvm
