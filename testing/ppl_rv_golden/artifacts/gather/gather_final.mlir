module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z13gather_kernelP4fp16S0_Pj(%arg0: memref<?xf16> {ppl.vname = "out"}, %arg1: memref<?xf16> {ppl.vname = "table"}, %arg2: memref<?xui32> {ppl.vname = "index"}) attributes {AddressAssigned = true, BlockNum = 1 : i32, GroupNum = 1 : i32, calc_liverange = true, category = "kernel", ori_name = "gather_kernel", tensor_idx_begin = 0 : i64} {
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ppl.tensorbe GLOBAL %arg1, %c1_i32, %c1_i32, %c16_i32, %c16_i32, %c256_i32, %c256_i32, %c16_i32, %c1_i32, %c0_i32 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_table"} : memref<?xf16>, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 2 : i32>
    %1 = ppl.tensorbe GLOBAL %arg2, %c1_i32, %c1_i32, %c4_i32, %c1_i32, %c4_i32, %c4_i32, %c1_i32, %c1_i32, %c0_i32 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_index"} : memref<?xui32>, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xui32, 2 : i32>
    %2 = ppl.tensorbe GLOBAL %arg0, %c1_i32, %c1_i32, %c4_i32, %c16_i32, %c64_i32, %c64_i32, %c16_i32, %c1_i32, %c0_i32 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out"} : memref<?xf16>, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 2 : i32>
    %3 = ppl.scalar %c0_i32 {reg_alloc = false} : i32 -> f16
    ppl.dma.h_gather %2, %0, %1, %3, %c0_i32 {loc = 0 : i32} : memref<?xf16, 2 : i32>, memref<?xf16, 2 : i32>, memref<?xui32, 2 : i32>, f16, i32
    return
  }
}
