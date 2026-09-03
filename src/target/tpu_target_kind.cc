/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file target/tpu_target_kind.cc
 * \brief Register the TileLang TPU target without modifying TVM sources.
 */

#include <tvm/ir/expr.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {

// The target kind belongs to the TileLang TPU backend, not to a copied TVM
// source file.  A few historic installations already have the old target
// registered by a patched TVM.  Register only when it is genuinely absent so
// registering this target kind is safe with either installation shape.
// ``mcpu`` is the canonical physical-chip selector; ``model`` remains a
// generic TVM annotation and is supplied by the common target options below.
void RegisterTPUTargetKindIfAbsent() {
  Optional<TargetKind> existing_target_kind = TargetKind::Get("tpu");
  if (existing_target_kind.defined()) {
    Map<String, String> options =
        TargetKindRegEntry::ListTargetKindOptions(existing_target_kind.value());
    ICHECK(options.count("mcpu"))
        << "An existing TVM target kind named 'tpu' does not accept -mcpu; "
        << "TileLang TPU requires -mcpu=<bm1690|sg2260e>. Rebuild against a "
        << "compatible TVM target registration.";
    if (!options.count("tpu-programming-model")) {
      // Old installations only registered the physical-chip option.  Extend
      // that compatible target kind with TileLang's internal model attribute
      // instead of replacing or patching TVM source files.
      TargetKindRegEntry::RegisterOrGet("tpu")
          .add_attr_option<String>("tpu-programming-model");
    }
    return;
  }

  TargetKindRegEntry::RegisterOrGet("tpu")
      .set_name()
      .set_default_device_type(kDLCPU)
      .add_attr_option<Array<String>>("keys")
      .add_attr_option<String>("tag")
      .add_attr_option<String>("device")
      .add_attr_option<String>("model")
      .add_attr_option<Array<String>>("libs")
      .add_attr_option<Target>("host")
      .add_attr_option<Integer>("from_device")
      .add_attr_option<Integer>("target_device_type")
      .add_attr_option<String>("mcpu")
      .add_attr_option<String>("tpu-programming-model")
      .add_attr_option<String>("march")
      .add_attr_option<Integer>("workspace-byte-alignment")
      .add_attr_option<Integer>("constants-byte-alignment")
      .set_default_keys({"cpu"});
}

struct TPUTargetKindRegistrar {
  TPUTargetKindRegistrar() { RegisterTPUTargetKindIfAbsent(); }
};

static TPUTargetKindRegistrar tpu_target_kind_registrar;

}  // namespace tvm
