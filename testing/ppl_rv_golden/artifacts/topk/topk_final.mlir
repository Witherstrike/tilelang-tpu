module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z11topk_kernelPfPiS_(%arg0: memref<?xf32> {ppl.vname = "out"}, %arg1: memref<?xi32> {ppl.vname = "out_index"}, %arg2: memref<?xf32> {ppl.vname = "in"}) attributes {AddressAssigned = true, BlockNum = 1 : i32, GroupNum = 1 : i32, calc_liverange = true, category = "kernel", ori_name = "topk_kernel", tensor_idx_begin = 0 : i64} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ppl.tensorbe GLOBAL %arg2, %c1_i32, %c1_i32, %c1_i32, %c16_i32, %c16_i32, %c16_i32, %c16_i32, %c1_i32, %c0_i32 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_in"} : memref<?xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf32, 2 : i32>
    %1 = ppl.tensorbe GLOBAL %arg0, %c1_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c1_i32, %c0_i32 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out"} : memref<?xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xf32, 2 : i32>
    %2 = ppl.tensorbe GLOBAL %arg1, %c1_i32, %c1_i32, %c1_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32, %c1_i32, %c0_i32 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out_index"} : memref<?xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32 -> memref<?xi32, 2 : i32>
    %3 = "ppl.none"() : () -> none
    ppl.hau.topK %1, %2, %0, %3, %c4_i32, %true {loc = 0 : i32} : memref<?xf32, 2 : i32>, memref<?xi32, 2 : i32>, memref<?xf32, 2 : i32>, none, i32, i1
    return
  }
}
