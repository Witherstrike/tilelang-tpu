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

#ifndef TVM_TL_BM1690_LMEM_H_
#define TVM_TL_BM1690_LMEM_H_

#include <tvm/tir/expr.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "tpu_chip_description.h"

namespace tvm {
namespace tl {
namespace bm1690 {

constexpr int64_t kLaneNum = 64;
constexpr int64_t kEuBytes = 64;
constexpr int64_t kBankNum = 16;
constexpr int64_t kBankSize = 16 * 1024;
constexpr int64_t kTensorAlignBytes = 64;

inline int64_t DivUp(int64_t value, int64_t factor) {
  return tl::DivUp(value, factor);
}

inline int64_t AlignUp(int64_t value, int64_t align) {
  return tl::AlignUp(value, align);
}

inline int64_t GetIntImmValue(const PrimExpr &expr, const char *context) {
  auto *imm = expr.as<IntImmNode>();
  ICHECK(imm) << context << " expects IntImm local tensor shapes";
  ICHECK_GT(imm->value, 0)
      << context << " expects positive local tensor shapes";
  return imm->value;
}

inline int64_t DTypeBytes(DataType dtype) {
  return tl::DTypeBytes(dtype);
}

inline std::vector<int64_t> NormalizeLocalShape(const Array<PrimExpr> &shape,
                                                const char *context) {
  if (shape.empty()) {
    return {1, 1, 1, 1};
  }
  if (shape.size() == 1) {
    return {1, 1, 1, GetIntImmValue(shape[0], context)};
  }
  if (shape.size() == 2) {
    return {1, GetIntImmValue(shape[0], context), 1,
            GetIntImmValue(shape[1], context)};
  }
  if (shape.size() == 3) {
    return {GetIntImmValue(shape[0], context),
            GetIntImmValue(shape[1], context), 1,
            GetIntImmValue(shape[2], context)};
  }
  if (shape.size() == 4) {
    return {GetIntImmValue(shape[0], context),
            GetIntImmValue(shape[1], context),
            GetIntImmValue(shape[2], context),
            GetIntImmValue(shape[3], context)};
  }
  LOG(FATAL) << context << " unsupported local tensor rank: "
             << shape.size();
  return {1, 1, 1, 1};
}

inline int64_t TpuAlignSizeBytesFromShape4(
    const std::vector<int64_t> &shape4, DataType dtype) {
  return tl::TpuAlignedSizeBytesFromShape4(kBM1690ChipDescription, shape4,
                                           dtype);
}

inline int64_t TpuAlignSizeBytes(const Array<PrimExpr> &shape, DataType dtype,
                                 const char *context) {
  return TpuAlignSizeBytesFromShape4(NormalizeLocalShape(shape, context),
                                     dtype);
}

} // namespace bm1690
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BM1690_LMEM_H_
