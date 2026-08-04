module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z16reduction_kernelP4fp16S0_(%arg0: memref<?xf16> {ppl.vname = "out"}, %arg1: memref<?xf16> {ppl.vname = "in"}) attributes {AddressAssigned = true, BlockNum = 1 : i32, GroupNum = 1 : i32, bank_bytes = 16384 : i64, bank_num = 16 : i64, calc_liverange = true, category = "kernel", dyn_block = false, lmem_size = 262144 : i64, ori_name = "reduction_kernel", tensor_idx_begin = 2 : i64, tensor_num = 2 : i64} {
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ppl.tensorbe GLOBAL %arg1, %c1_i32, %c8_i32, %c1_i32, %c16_i32, %c128_i32, %c16_i32, %c16_i32, %c1_i32, %c0_i32 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_in"} : memref<?xf16>, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 2 : i32>
    %1 = ppl.tensorbe GLOBAL %arg0, %c1_i32, %c8_i32, %c1_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32, %c1_i32, %c0_i32 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out"} : memref<?xf16>, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 2 : i32>
    %2 = "ppl.none"() : () -> none
    %3 = ppl.tensorbe LOCAL %2, %c1_i32, %c8_i32, %c1_i32, %c16_i32, %c32_i32, %c32_i32, %c16_i32, %c1_i32, %c64_i32 {address = 16384 : i64, align_mode = 1 : i64, bank_conflict = [1], idx = 0 : i32, live_range = [0, 2], ppl.vname = "local_in", size = 64 : i64} : none, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 3 : i32>
    %4 = ppl.tensorbe LOCAL %2, %c1_i32, %c8_i32, %c1_i32, %c1_i32, %c32_i32, %c32_i32, %c1_i32, %c1_i32, %c64_i32 {address = 0 : i64, align_mode = 1 : i64, bank_conflict = [0], idx = 1 : i32, live_range = [1, 3], ppl.vname = "local_out", size = 64 : i64} : none, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 3 : i32>
    ppl.dma.load %3, %0 {loc = 0 : i32} : memref<?xf16, 3 : i32>, memref<?xf16, 2 : i32>
    ppl.tiu.reduce 6 %4, %3, %false {loc = 1 : i32} : memref<?xf16, 3 : i32>, memref<?xf16, 3 : i32>, i1
    ppl.dma.store %1, %4 {loc = 2 : i32} : memref<?xf16, 2 : i32>, memref<?xf16, 3 : i32>
    return
  }
}
