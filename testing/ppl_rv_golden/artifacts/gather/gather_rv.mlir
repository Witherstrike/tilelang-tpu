module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z13gather_kernelP4fp16S0_Pj(%arg0: memref<?xf16> {ppl.vname = "out"}, %arg1: memref<?xf16> {ppl.vname = "table"}, %arg2: memref<?xui32> {ppl.vname = "index"}) attributes {AddressAssigned = true, BlockNum = 1 : i32, GroupNum = 1 : i32, calc_liverange = true, category = "kernel", ori_name = "gather_kernel", tensor_idx_begin = 0 : i64} {
    %c33_i32 = arith.constant 33 : i32
    %c32_i32 = arith.constant 32 : i32
    %c34_i32 = arith.constant 34 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<?xf16> to i64
    ppl_rv.reg_wgaddr %c33_i32, %0, %c1_i32 {dtype = 2 : i32} : i32, i64, i32
    ppl_rv.reg_wgshape %c33_i32, %c1_i32, %c1_i32, %c16_i32, %c16_i32 : i32, i32, i32, i32, i32
    %1 = builtin.unrealized_conversion_cast %arg2 : memref<?xui32> to i64
    ppl_rv.reg_wgaddr %c32_i32, %1, %c1_i32 {dtype = 9 : i32} : i32, i64, i32
    ppl_rv.reg_wgshape %c32_i32, %c1_i32, %c1_i32, %c4_i32, %c1_i32 : i32, i32, i32, i32, i32
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<?xf16> to i64
    ppl_rv.reg_wgaddr %c34_i32, %2, %c1_i32 {dtype = 2 : i32} : i32, i64, i32
    ppl_rv.reg_wgshape %c34_i32, %c1_i32, %c1_i32, %c4_i32, %c16_i32 : i32, i32, i32, i32, i32
    %3 = ppl_rv.scalar %c0_i32 {reg_alloc = false} : i32 -> f16
    ppl_rv.reg_dma_idx %c0_i32, %3 : i32, f16
    ppl_rv.dma_gather_h %c34_i32, %c33_i32, %c32_i32, %c0_i32 : i32, i32, i32, i32
    return
  }
}
