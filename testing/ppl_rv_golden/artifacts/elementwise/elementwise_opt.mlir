module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z18elementwise_kernelP4fp16S0_S0_(%arg0: memref<?xf16> {ppl.vname = "out"}, %arg1: memref<?xf16> {ppl.vname = "lhs"}, %arg2: memref<?xf16> {ppl.vname = "rhs"}) attributes {BlockNum = 1 : i32, GroupNum = 1 : i32, category = "kernel", ori_name = "elementwise_kernel"} {
    %false = arith.constant false
    %cst = arith.constant 5.000000e-01 : f32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ppl.shape %c1_i32, %c8_i32, %c1_i32, %c16_i32 {ppl.vname = "shape", struct = "dim4"} : i32, i32, i32, i32 -> tensor<1x4xi32>
    %1 = ppl.tensorfe GLOBAL %0, %arg1 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_lhs"} : tensor<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %2 = ppl.tensorfe GLOBAL %0, %arg2 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_rhs"} : tensor<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %3 = ppl.tensorfe GLOBAL %0, %arg0 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out"} : tensor<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %4 = "ppl.none"() : () -> none
    %5 = ppl.tensorfe LOCAL %0, %4 {address = -1 : i64, align_mode = 1 : i64, ppl.vname = "l"} : tensor<1x4xi32>, none -> memref<?xf16>
    %6 = ppl.tensorfe LOCAL %0, %4 {address = -1 : i64, align_mode = 1 : i64, ppl.vname = "r"} : tensor<1x4xi32>, none -> memref<?xf16>
    %7 = ppl.tensorfe LOCAL %0, %4 {address = -1 : i64, align_mode = 1 : i64, ppl.vname = "sum"} : tensor<1x4xi32>, none -> memref<?xf16>
    %8 = ppl.tensorfe LOCAL %0, %4 {address = -1 : i64, align_mode = 1 : i64, ppl.vname = "result"} : tensor<1x4xi32>, none -> memref<?xf16>
    ppl.dma.load %5, %1 : memref<?xf16>, memref<?xf16>
    ppl.dma.load %6, %2 : memref<?xf16>, memref<?xf16>
    ppl.tiu.arith ADD %7, %5, %6, %false : memref<?xf16>, memref<?xf16>, memref<?xf16>, i1
    %9 = ppl.scalar %cst : f32 -> f16
    ppl.tiu.arith MUL %8, %7, %9, %false : memref<?xf16>, memref<?xf16>, f16, i1
    ppl.dma.store %3, %8 : memref<?xf16>, memref<?xf16>
    return
  }
}
