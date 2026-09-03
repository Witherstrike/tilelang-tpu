# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
from tvm import tir, IRModule
from tvm.target import Target
import tilelang


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    # Bind the target device information to the module
    """
    disable pass(TODO: need verify):
        layout inference
        lower Tile op
        safememory
    """
    mod = tir.transform.BindTarget(target)(mod)

    # Legalize the frontend IR to make it compatible with TVM
    mod = tilelang.transform.FrontendLegalize()(mod)
    # Simplify the IR expressions
    mod = tir.transform.Simplify()(mod)
    # Infer memory layouts for fragments and shared memory
    # mod = tilelang.transform.LayoutInference()(mod)
    # Lower high-level tile operations to low-level operations
    # mod = tilelang.transform.LowerTileOp()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    # mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    mod = tir.transform.Simplify()(mod)
    # Try to vectorize loop with dynamic shape
    # mod = tilelang.transform.LoopVectorizeDynamic()(mod)

    return mod


def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    # which may be introduced by the LegalizeSafeMemoryAccess
    if target.arch == "sm_90":
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tilelang.transform.MultiVersionBuffer()(mod)
        mod = tilelang.transform.WarpSpecialized()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        mod = tir.transform.LowerOpaqueBlock()(mod)
        mod = tilelang.transform.MergeIfStmt()(mod)
        mod = tilelang.transform.RewriteWgmmaSync()(mod)
        mod = tilelang.transform.InjectFenceProxy()(mod)
    else:
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        mod = tilelang.transform.MergeIfStmt()(mod)

    # TODO(lei): may need a pass to fuse the if-then-else in the
    # pipeline loop when we meet dynamic branch.
    mod = tir.transform.LowerOpaqueBlock()(mod)
    # no need flatten buffer due to tpu has 4D tensor
    # mod = tir.transform.FlattenBuffer()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop()(mod)
    # StorageRewrite assumes FlattenBuffer/StorageFlatten-style IR, where
    # Allocate/DeclBuffer pairs have already been normalized. The TPU path
    # intentionally keeps multi-dimensional buffers for downstream region-based
    # codegen, so running StorageRewrite after LowerOpaqueBlock introduces
    # duplicate declarations for the same buffer var.
    if target.kind.name != "tpu":
        mod = tir.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.AddressAssign()(mod)
    return mod
