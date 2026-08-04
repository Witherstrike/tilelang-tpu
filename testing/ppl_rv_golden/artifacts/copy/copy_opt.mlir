module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z11copy_kernelP4fp16S0_(%arg0: memref<?xf16> {ppl.vname = "out"}, %arg1: memref<?xf16> {ppl.vname = "in"}) attributes {BlockNum = 1 : i32, GroupNum = 1 : i32, category = "kernel", ori_name = "copy_kernel"} {
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ppl.shape %c1_i32, %c8_i32, %c1_i32, %c16_i32 {ppl.vname = "shape", struct = "dim4"} : i32, i32, i32, i32 -> tensor<1x4xi32>
    %1 = ppl.tensorfe GLOBAL %0, %arg1 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_in"} : tensor<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %2 = ppl.tensorfe GLOBAL %0, %arg0 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out"} : tensor<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %3 = "ppl.none"() : () -> none
    %4 = ppl.tensorfe LOCAL %0, %3 {address = -1 : i64, align_mode = 1 : i64, ppl.vname = "local"} : tensor<1x4xi32>, none -> memref<?xf16>
    ppl.dma.load %4, %1 : memref<?xf16>, memref<?xf16>
    ppl.dma.store %2, %4 : memref<?xf16>, memref<?xf16>
    return
  }
}
