// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_TRANSFORM_RV_IR_H_
#define TVM_TL_TRANSFORM_RV_IR_H_

namespace tvm {
namespace tl {
namespace rv {

constexpr const char *kTensorView = "ppl.rv.tensor_view";
constexpr const char *kScalar = "ppl.rv.scalar";
constexpr const char *kLegalizedAttr = "tir.tpu.rv.legalized";
constexpr const char *kSchemaVersionAttr = "tir.tpu.rv.schema_version";
constexpr const char *kChipAttr = "tir.tpu.chip";
constexpr const char *kDeviceModeAttr = "tir.tpu.device_mode";

enum class RegisterClass : int { kControl = 0, kTensor = 1, kGlobal = 2 };
enum class MemoryScope : int { kLocal = 0, kGlobal = 1 };
enum class LayoutKind : int {
  kHwAligned = 0,
  kContinuous = 1,
  kRowAligned = 2,
  kFree = 3,
  kMatrix = 4,
};

} // namespace rv
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_RV_IR_H_
