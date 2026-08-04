module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z16reduction_kernelP4fp16S0_(%arg0: memref<?xf16> {ppl.vname = "out"}, %arg1: memref<?xf16> {ppl.vname = "in"}) attributes {BlockNum = 1 : i32, GroupNum = 1 : i32, category = "kernel", ori_name = "reduction_kernel"} {
    %false = arith.constant false
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ppl.shape %c1_i32, %c8_i32, %c1_i32, %c16_i32 {ppl.vname = "in_shape", struct = "dim4"} : i32, i32, i32, i32 -> tensor<1x4xi32>
    %1 = ppl.shape %c1_i32, %c8_i32, %c1_i32, %c1_i32 {ppl.vname = "out_shape", struct = "dim4"} : i32, i32, i32, i32 -> tensor<1x4xi32>
    %2 = ppl.tensorfe GLOBAL %0, %arg1 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_in"} : tensor<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %3 = ppl.tensorfe GLOBAL %1, %arg0 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out"} : tensor<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %4 = "ppl.none"() : () -> none
    %5 = ppl.tensorfe LOCAL %0, %4 {address = -1 : i64, align_mode = 1 : i64, ppl.vname = "local_in"} : tensor<1x4xi32>, none -> memref<?xf16>
    %6 = ppl.tensorfe LOCAL %1, %4 {address = -1 : i64, align_mode = 1 : i64, ppl.vname = "local_out"} : tensor<1x4xi32>, none -> memref<?xf16>
    ppl.dma.load %5, %2 : memref<?xf16>, memref<?xf16>
    ppl.tiu.reduce 6 %6, %5, %false : memref<?xf16>, memref<?xf16>, i1
    ppl.dma.store %3, %6 : memref<?xf16>, memref<?xf16>
    return
  }
}
