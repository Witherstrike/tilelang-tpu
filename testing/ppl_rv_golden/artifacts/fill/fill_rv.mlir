module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z11fill_kernelP4fp16(%arg0: memref<?xf16> {ppl.vname = "out"}) attributes {AddressAssigned = true, BlockNum = 1 : i32, GroupNum = 1 : i32, bank_bytes = 16384 : i64, bank_num = 16 : i64, calc_liverange = true, category = "kernel", dyn_block = false, lmem_size = 262144 : i64, ori_name = "fill_kernel", tensor_idx_begin = 1 : i64, tensor_num = 1 : i64} {
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.500000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<?xf16> to i64
    ppl_rv.reg_wgaddr %c32_i32, %0, %c1_i32 {dtype = 2 : i32} : i32, i64, i32
    ppl_rv.reg_wgshape %c32_i32, %c1_i32, %c8_i32, %c1_i32, %c16_i32 : i32, i32, i32, i32, i32
    ppl_rv.reg_wladdr %c8_i32, %c0_i32, %c0_i32 {dtype = 2 : i32} : i32, i32, i32
    ppl_rv.reg_wlshape %c8_i32, %c1_i32, %c8_i32, %c1_i32, %c16_i32 : i32, i32, i32, i32, i32
    %1 = ppl_rv.scalar %cst : f32 -> f16
    %2 = ppl_rv.reg_wscalar %1 {dtype = 2 : i32, reg = 1 : i32} : f16 -> i32
    ppl_rv.tiu_set_c %c8_i32, %2 {loc = 0 : i32} : i32, i32
    ppl_rv.dmastore %c32_i32, %c8_i32 {loc = 1 : i32} : i32, i32
    return
  }
}
