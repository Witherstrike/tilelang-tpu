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

#ifndef TVM_TL_TPU_CHIP_DESCRIPTION_H_
#define TVM_TL_TPU_CHIP_DESCRIPTION_H_

#include <tvm/runtime/logging.h>
#include <tvm/runtime/data_type.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

struct ChipDescription {
  const char *name;
  int64_t lane_num;
  int64_t eu_bytes;
  int64_t bank_num;
  int64_t bank_size;
  int64_t tensor_align_bytes;
};

inline constexpr ChipDescription kBM1690ChipDescription{
    "bm1690", 64, 64, 16, 16 * 1024, 64};
inline constexpr ChipDescription kSG2260EChipDescription{
    "sg2260e", 64, 64, 16, 16 * 1024, 64};

inline const ChipDescription &GetChipDescription(const std::string &chip) {
  if (chip.empty() || chip == "bm1690") {
    return kBM1690ChipDescription;
  }
  if (chip == "sg2260e") {
    return kSG2260EChipDescription;
  }
  LOG(FATAL) << "Unsupported TPU chip for local-memory allocation: " << chip;
  return kBM1690ChipDescription;
}

inline int64_t DivUp(int64_t value, int64_t factor) {
  ICHECK_GT(factor, 0);
  return (value + factor - 1) / factor;
}

inline int64_t AlignUp(int64_t value, int64_t align) {
  ICHECK_GT(align, 0);
  return DivUp(value, align) * align;
}

inline int64_t DTypeBytes(DataType dtype) {
  const int64_t bits = static_cast<int64_t>(dtype.bits()) * dtype.lanes();
  return std::max<int64_t>(1, DivUp(bits, 8));
}

inline int64_t TpuAlignedSizeBytesFromShape4(
    const ChipDescription &chip, const std::vector<int64_t> &shape4,
    DataType dtype) {
  ICHECK_EQ(shape4.size(), 4U);
  const int64_t dtype_bytes = DTypeBytes(dtype);
  const int64_t eu_num = std::max<int64_t>(1, chip.eu_bytes / dtype_bytes);
  const int64_t stride_c = AlignUp(shape4[2] * shape4[3], eu_num);
  const int64_t lane_groups = DivUp(shape4[1], chip.lane_num);
  const int64_t bytes =
      shape4[0] * lane_groups * stride_c * dtype_bytes;
  return AlignUp(std::max<int64_t>(bytes, chip.tensor_align_bytes),
                 chip.tensor_align_bytes);
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TPU_CHIP_DESCRIPTION_H_
