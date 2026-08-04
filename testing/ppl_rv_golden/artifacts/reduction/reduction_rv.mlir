module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z16reduction_kernelP4fp16S0_(%arg0: memref<?xf16> {ppl.vname = "out"}, %arg1: memref<?xf16> {ppl.vname = "in"}) attributes {AddressAssigned = true, BlockNum = 1 : i32, GroupNum = 1 : i32, bank_bytes = 16384 : i64, bank_num = 16 : i64, calc_liverange = true, category = "kernel", dyn_block = false, lmem_size = 262144 : i64, ori_name = "reduction_kernel", tensor_idx_begin = 2 : i64, tensor_num = 2 : i64} {
    %c32_i32 = arith.constant 32 : i32
    %c33_i32 = arith.constant 33 : i32
    %c8_i32 = arith.constant 8 : i32
    %c9_i32 = arith.constant 9 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c16_i32 = arith.constant 16 : i32
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<?xf16> to i64
    ppl_rv.reg_wgaddr %c32_i32, %0, %c1_i32 {dtype = 2 : i32} : i32, i64, i32
    ppl_rv.reg_wgshape %c32_i32, %c1_i32, %c8_i32, %c1_i32, %c16_i32 : i32, i32, i32, i32, i32
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<?xf16> to i64
    ppl_rv.reg_wgaddr %c33_i32, %1, %c1_i32 {dtype = 2 : i32} : i32, i64, i32
    ppl_rv.reg_wgshape %c33_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : i32, i32, i32, i32, i32
    ppl_rv.reg_wladdr %c8_i32, %c16384_i32, %c0_i32 {dtype = 2 : i32} : i32, i32, i32
    ppl_rv.reg_wlshape %c8_i32, %c1_i32, %c8_i32, %c1_i32, %c16_i32 : i32, i32, i32, i32, i32
    ppl_rv.reg_wladdr %c9_i32, %c0_i32, %c0_i32 {dtype = 2 : i32} : i32, i32, i32
    ppl_rv.reg_wlshape %c9_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : i32, i32, i32, i32, i32
    ppl_rv.dmaload %c8_i32, %c32_i32 {loc = 0 : i32} : i32, i32
    ppl_rv.reg_satu %false, %false : i1, i1
    ppl_rv.tiu_reduce SUM %c9_i32, %c8_i32 : i32, i32
    ppl_rv.dmastore %c33_i32, %c9_i32 {loc = 2 : i32} : i32, i32
    return
  }
}
