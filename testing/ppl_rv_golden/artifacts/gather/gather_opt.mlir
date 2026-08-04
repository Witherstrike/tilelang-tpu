module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z13gather_kernelP4fp16S0_Pj(%arg0: memref<?xf16> {ppl.vname = "out"}, %arg1: memref<?xf16> {ppl.vname = "table"}, %arg2: memref<?xui32> {ppl.vname = "index"}) attributes {BlockNum = 1 : i32, GroupNum = 1 : i32, category = "kernel", ori_name = "gather_kernel"} {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ppl.shape %c1_i32, %c1_i32, %c16_i32, %c16_i32 {ppl.vname = "table_shape", struct = "dim4"} : i32, i32, i32, i32 -> tensor<1x4xi32>
    %1 = ppl.shape %c1_i32, %c1_i32, %c4_i32, %c1_i32 {ppl.vname = "index_shape", struct = "dim4"} : i32, i32, i32, i32 -> tensor<1x4xi32>
    %2 = ppl.shape %c1_i32, %c1_i32, %c4_i32, %c16_i32 {ppl.vname = "out_shape", struct = "dim4"} : i32, i32, i32, i32 -> tensor<1x4xi32>
    %3 = ppl.tensorfe GLOBAL %0, %arg1 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_table"} : tensor<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %4 = ppl.tensorfe GLOBAL %1, %arg2 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_index"} : tensor<1x4xi32>, memref<?xui32> -> memref<?xui32>
    %5 = ppl.tensorfe GLOBAL %2, %arg0 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out"} : tensor<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %6 = ppl.scalar %c0_i32 {reg_alloc = false} : i32 -> f16
    ppl.dma.h_gather %5, %3, %4, %6, %c0_i32 : memref<?xf16>, memref<?xf16>, memref<?xui32>, f16, i32
    return
  }
}
