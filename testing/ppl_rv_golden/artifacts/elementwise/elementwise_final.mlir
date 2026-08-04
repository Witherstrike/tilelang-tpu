module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z18elementwise_kernelP4fp16S0_S0_(%arg0: memref<?xf16> {ppl.vname = "out"}, %arg1: memref<?xf16> {ppl.vname = "lhs"}, %arg2: memref<?xf16> {ppl.vname = "rhs"}) attributes {AddressAssigned = true, BlockNum = 1 : i32, GroupNum = 1 : i32, bank_bytes = 16384 : i64, bank_num = 16 : i64, calc_liverange = true, category = "kernel", dyn_block = false, lmem_size = 262144 : i64, ori_name = "elementwise_kernel", tensor_idx_begin = 4 : i64, tensor_num = 4 : i64} {
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %cst = arith.constant 5.000000e-01 : f32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ppl.tensorbe GLOBAL %arg1, %c1_i32, %c8_i32, %c1_i32, %c16_i32, %c128_i32, %c16_i32, %c16_i32, %c1_i32, %c0_i32 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_lhs"} : memref<?xf16>, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 2 : i32>
    %1 = ppl.tensorbe GLOBAL %arg2, %c1_i32, %c8_i32, %c1_i32, %c16_i32, %c128_i32, %c16_i32, %c16_i32, %c1_i32, %c0_i32 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_rhs"} : memref<?xf16>, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 2 : i32>
    %2 = ppl.tensorbe GLOBAL %arg0, %c1_i32, %c8_i32, %c1_i32, %c16_i32, %c128_i32, %c16_i32, %c16_i32, %c1_i32, %c0_i32 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out"} : memref<?xf16>, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 2 : i32>
    %3 = "ppl.none"() : () -> none
    %4 = ppl.tensorbe LOCAL %3, %c1_i32, %c8_i32, %c1_i32, %c16_i32, %c32_i32, %c32_i32, %c16_i32, %c1_i32, %c64_i32 {address = 32768 : i64, align_mode = 1 : i64, bank_conflict = [1, 2], idx = 0 : i32, live_range = [0, 3], ppl.vname = "l", size = 64 : i64} : none, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 3 : i32>
    %5 = ppl.tensorbe LOCAL %3, %c1_i32, %c8_i32, %c1_i32, %c16_i32, %c32_i32, %c32_i32, %c16_i32, %c1_i32, %c64_i32 {address = 0 : i64, align_mode = 1 : i64, bank_conflict = [0, 2], idx = 1 : i32, live_range = [1, 3], ppl.vname = "r", size = 64 : i64} : none, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 3 : i32>
    %6 = ppl.tensorbe LOCAL %3, %c1_i32, %c8_i32, %c1_i32, %c16_i32, %c32_i32, %c32_i32, %c16_i32, %c1_i32, %c64_i32 {address = 16384 : i64, align_mode = 1 : i64, bank_conflict = [0, 1, 3], idx = 2 : i32, live_range = [2, 4], ppl.vname = "sum", size = 64 : i64} : none, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 3 : i32>
    %7 = ppl.tensorbe LOCAL %3, %c1_i32, %c8_i32, %c1_i32, %c16_i32, %c32_i32, %c32_i32, %c16_i32, %c1_i32, %c64_i32 {address = 0 : i64, align_mode = 1 : i64, bank_conflict = [2], idx = 3 : i32, live_range = [3, 5], ppl.vname = "result", size = 64 : i64} : none, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf16, 3 : i32>
    ppl.dma.load %4, %0 {loc = 0 : i32} : memref<?xf16, 3 : i32>, memref<?xf16, 2 : i32>
    ppl.dma.load %5, %1 {loc = 1 : i32} : memref<?xf16, 3 : i32>, memref<?xf16, 2 : i32>
    ppl.tiu.arith ADD %6, %4, %5, %false {bc_checked = true, loc = 2 : i32} : memref<?xf16, 3 : i32>, memref<?xf16, 3 : i32>, memref<?xf16, 3 : i32>, i1
    %8 = ppl.scalar %cst : f32 -> f16
    ppl.tiu.arith MUL %7, %6, %8, %false {bc_checked = true, loc = 3 : i32} : memref<?xf16, 3 : i32>, memref<?xf16, 3 : i32>, f16, i1
    ppl.dma.store %2, %7 {loc = 4 : i32} : memref<?xf16, 2 : i32>, memref<?xf16, 3 : i32>
    return
  }
}
