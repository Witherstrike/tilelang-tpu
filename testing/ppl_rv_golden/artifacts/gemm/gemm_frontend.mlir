module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z11gemm_kernelP4fp16S0_S0_(%arg0: memref<?xf16> {ppl.vname = "out"}, %arg1: memref<?xf16> {ppl.vname = "lhs"}, %arg2: memref<?xf16> {ppl.vname = "rhs"}) attributes {BlockNum = 1 : i32, GroupNum = 1 : i32, category = "kernel", ori_name = "gemm_kernel"} {
    %c0_i32 = arith.constant {ppl.vname = "bias"} 0 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %false = arith.constant false
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ppl.shape %c1_i32, %c16_i32, %c1_i32, %c16_i32 {ppl.vname = "lhs_shape", struct = "dim4"} : i32, i32, i32, i32 -> memref<1x4xi32>
    %1 = ppl.shape %c1_i32, %c16_i32, %c1_i32, %c16_i32 {ppl.vname = "rhs_shape", struct = "dim4"} : i32, i32, i32, i32 -> memref<1x4xi32>
    %2 = ppl.shape %c1_i32, %c16_i32, %c1_i32, %c16_i32 {ppl.vname = "out_shape", struct = "dim4"} : i32, i32, i32, i32 -> memref<1x4xi32>
    %3 = ppl.tensorfe GLOBAL %0, %arg1 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_lhs"} : memref<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %4 = ppl.tensorfe GLOBAL %1, %arg2 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_rhs"} : memref<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %5 = ppl.tensorfe GLOBAL %2, %arg0 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out"} : memref<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %6 = "ppl.none"() : () -> none
    %7 = ppl.tensorfe LOCAL %0, %6 {address = -1 : i64, align_mode = 1 : i64, ppl.vname = "l"} : memref<1x4xi32>, none -> memref<?xf16>
    %8 = ppl.tensorfe LOCAL %1, %6 {address = -1 : i64, align_mode = 1 : i64, ppl.vname = "r"} : memref<1x4xi32>, none -> memref<?xf16>
    %9 = ppl.tensorfe LOCAL %2, %6 {address = -1 : i64, align_mode = 1 : i64, ppl.vname = "result"} : memref<1x4xi32>, none -> memref<?xf16>
    ppl.dma.load %7, %3 : memref<?xf16>, memref<?xf16>
    ppl.dma.load %8, %4 : memref<?xf16>, memref<?xf16>
    ppl.tiu.fmm2.nn %9, %7, %8, %c0_i32, %false, %c0_i32, %false, %false, %cst : memref<?xf16>, memref<?xf16>, memref<?xf16>, i32, i1, i32, i1, i1, f32
    ppl.dma.store %5, %9 : memref<?xf16>, memref<?xf16>
    return
  }
}
