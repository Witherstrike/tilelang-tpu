/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file distributed
 * with this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0.
 */

/*! \file target/codegen_tpukernel.cc
 *  \brief TPU-Kernel instruction selection for backend-neutral TPU ops.
 */

#include "codegen_tpu_common.h"

#include <tvm/runtime/logging.h>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace tvm {
namespace codegen {

namespace {

const char *TPUKernelDTypeName(DataType dtype) {
  if (dtype == DataType::Float(32)) {
    return "DT_FP32";
  }
  if (dtype == DataType::Float(16)) {
    return "DT_FP16";
  }
  if (dtype == DataType::BFloat(16)) {
    return "DT_BFP16";
  }
  LOG(FATAL) << "TPU-Kernel floating-point op does not support dtype " << dtype;
  return "DT_FP32";
}

std::string FloatingLiteral(double value) {
  if (std::isnan(value)) {
    return "(0.0f / 0.0f)";
  }
  if (std::isinf(value)) {
    return value > 0 ? "(1.0f / 0.0f)" : "(-1.0f / 0.0f)";
  }
  std::ostringstream os;
  os << std::scientific << value << 'f';
  return os.str();
}

} // namespace

void CodeGenTileLangTPU::EmitTPUKernelCopy(
    const std::string &src, bool src_is_global, const std::string &src_dtype,
    const std::string &dst, bool dst_is_global, const std::string &dst_dtype) {
  PrintIndent();
  if (src_dtype != dst_dtype) {
    ICHECK(!src_is_global && !dst_is_global)
        << "TPU copy-and-convert currently requires two local tensors";
    ICHECK(!((src_dtype == "DT_FP16" && dst_dtype == "DT_BFP16") ||
             (src_dtype == "DT_BFP16" && dst_dtype == "DT_FP16")))
        << "TPU-Kernel tpu_bdc_cast does not support direct FP16/BF16 "
           "conversion";
    stream << "tpu_bdc_cast(" << dst << ".addr, " << src << ".addr, &" << dst
           << ".shape, (" << dst << ".default_stride ? NULL : &" << dst
           << ".stride), (" << src << ".default_stride ? NULL : &" << src
           << ".stride), " << dst_dtype << ", " << src_dtype
           << ", RM_HALF_TO_EVEN);\n";
    return;
  }

  const char *instruction = nullptr;
  if (src_is_global && !dst_is_global) {
    instruction = "tpu_gdma_cpy_S2L";
  } else if (!src_is_global && dst_is_global) {
    instruction = "tpu_gdma_cpy_L2S";
  } else if (src_is_global && dst_is_global) {
    instruction = "tpu_gdma_cpy_S2S";
  } else {
    instruction = "tpu_bdc_cpy";
  }
  stream << instruction << "(" << dst << ".addr, " << src << ".addr, &"
         << dst << ".shape, (" << dst << ".default_stride ? NULL : &" << dst
         << ".stride), (" << src << ".default_stride ? NULL : &" << src
         << ".stride), " << src_dtype << ");\n";
}

void CodeGenTileLangTPU::EmitTPUKernelFill(const std::string &dst,
                                            DataType dtype, double value) {
  const char *scalar_field = nullptr;
  if (dtype == DataType::Float(16)) {
    scalar_field = "f16";
  } else if (dtype == DataType::Float(32)) {
    scalar_field = "f32";
  } else if (dtype == DataType::BFloat(16)) {
    scalar_field = "bf16";
  } else {
    LOG(FATAL) << "TPU-Kernel fill does not support dtype " << dtype;
  }
  const char *dtype_name = TPUKernelDTypeName(dtype);

  PrintIndent();
  stream << "{\n";
  PrintIndent();
  stream << "scalar_t " << dst << "_scalar_" << scalar_field
         << " = {.f32 = " << FloatingLiteral(value) << "};\n";
  if (dtype != DataType::Float(32)) {
    PrintIndent();
    stream << dst << "_scalar_" << scalar_field << " = tpu_cast(" << dst
           << "_scalar_" << scalar_field << ", " << dtype_name
           << ", DT_FP32, RM_HALF_TO_EVEN);\n";
  }
  PrintIndent();
  stream << "tpu_bdc_set_C(" << dst << ".addr, " << dst << "_scalar_"
         << scalar_field << ", &" << dst << ".shape, (" << dst
         << ".default_stride ? NULL : &" << dst << ".stride), " << dtype_name
         << ");\n";
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangTPU::EmitTPUKernelGemm(
    const std::string &a, const std::string &b, const std::string &c,
    DataType a_dtype, DataType b_dtype, DataType c_dtype, bool transpose_a,
    bool transpose_b, bool accumulate, int64_t m, int64_t n, int64_t k) {
  ICHECK(!transpose_a)
      << "TileLang TPU GEMM does not yet support transpose_A=true";
  ICHECK(a_dtype == b_dtype)
      << "TPU-Kernel GEMM requires matching input dtypes, got " << a_dtype
      << " and " << b_dtype;
  ICHECK(a_dtype == DataType::Float(16) || a_dtype == DataType::BFloat(16))
      << "TPU-Kernel GEMM currently requires FP16 or BF16 inputs, got "
      << a_dtype;
  ICHECK(c_dtype == DataType::Float(32) ||
         (!accumulate && c_dtype == a_dtype))
      << "TPU-Kernel NN accumulation requires an FP32 C tile; the legacy "
         "overwrite forms additionally support C matching A/B";
  ICHECK(!transpose_b || !accumulate)
      << "TPU-Kernel has no accumulating right-transpose GEMM instruction; "
         "use accumulate=False or materialize the transpose";
  const char *input_dtype = TPUKernelDTypeName(a_dtype);
  const char *output_dtype = TPUKernelDTypeName(c_dtype);

  PrintIndent();
  if (!transpose_b) {
    stream << "tpu_bdc_fp_mm(" << c << ".addr, " << a << ".addr, " << b
           << ".addr, " << m << ", " << k << ", " << n << ", "
           << output_dtype << ", " << input_dtype << ", "
           << (accumulate ? "true" : "false") << ");\n";
  } else {
    stream << "tpu_bdc_fp_mm_R_trans(" << c << ".addr, " << a << ".addr, "
           << b << ".addr, " << m << ", " << k << ", " << n << ", "
           << output_dtype << ", " << input_dtype << ");\n";
  }
}

void CodeGenTileLangTPU::EmitTPUKernelElementwise(
    const std::string &instruction, const std::string &dst,
    const std::string &src0, const std::string &src1, const std::string &dtype,
    const std::string &src1_stride) {
  PrintIndent();
  stream << instruction << "(" << dst << ".addr, " << src0 << ".addr, "
         << src1 << ".addr, &" << dst << ".shape, (" << dst
         << ".default_stride ? NULL : &" << dst << ".stride), (" << src0
         << ".default_stride ? NULL : &" << src0 << ".stride), "
         << src1_stride << dtype << ");\n";
}

} // namespace codegen
} // namespace tvm
