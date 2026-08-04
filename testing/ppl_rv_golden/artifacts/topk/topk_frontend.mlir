module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z11topk_kernelPfPiS_(%arg0: memref<?xf32> {ppl.vname = "out"}, %arg1: memref<?xi32> {ppl.vname = "out_index"}, %arg2: memref<?xf32> {ppl.vname = "in"}) attributes {BlockNum = 1 : i32, GroupNum = 1 : i32, category = "kernel", ori_name = "topk_kernel"} {
    %true = arith.constant true
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ppl.shape %c1_i32, %c1_i32, %c1_i32, %c16_i32 {ppl.vname = "in_shape", struct = "dim4"} : i32, i32, i32, i32 -> memref<1x4xi32>
    %1 = ppl.shape %c1_i32, %c1_i32, %c1_i32, %c4_i32 {ppl.vname = "out_shape", struct = "dim4"} : i32, i32, i32, i32 -> memref<1x4xi32>
    %2 = ppl.tensorfe GLOBAL %0, %arg2 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_in"} : memref<1x4xi32>, memref<?xf32> -> memref<?xf32>
    %3 = ppl.tensorfe GLOBAL %1, %arg0 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out"} : memref<1x4xi32>, memref<?xf32> -> memref<?xf32>
    %4 = ppl.tensorfe GLOBAL %1, %arg1 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out_index"} : memref<1x4xi32>, memref<?xi32> -> memref<?xi32>
    %5 = "ppl.none"() : () -> none
    ppl.hau.topK %3, %4, %2, %5, %c4_i32, %true : memref<?xf32>, memref<?xi32>, memref<?xf32>, none, i32, i1
    return
  }
}
