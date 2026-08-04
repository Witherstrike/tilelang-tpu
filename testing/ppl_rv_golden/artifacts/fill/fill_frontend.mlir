module attributes {module.chip = "tpub_7_1_e_rv"} {
  func.func @_Z11fill_kernelP4fp16(%arg0: memref<?xf16> {ppl.vname = "out"}) attributes {BlockNum = 1 : i32, GroupNum = 1 : i32, category = "kernel", ori_name = "fill_kernel"} {
    %cst = arith.constant 1.500000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ppl.shape %c1_i32, %c8_i32, %c1_i32, %c16_i32 {ppl.vname = "shape", struct = "dim4"} : i32, i32, i32, i32 -> memref<1x4xi32>
    %1 = ppl.tensorfe GLOBAL %0, %arg0 {address = -1 : i64, align_mode = 0 : i64, ppl.vname = "g_out"} : memref<1x4xi32>, memref<?xf16> -> memref<?xf16>
    %2 = "ppl.none"() : () -> none
    %3 = ppl.tensorfe LOCAL %0, %2 {address = -1 : i64, align_mode = 1 : i64, ppl.vname = "local"} : memref<1x4xi32>, none -> memref<?xf16>
    ppl.tiu.set_C %3, %cst : memref<?xf16>, f32
    ppl.dma.store %1, %3 : memref<?xf16>, memref<?xf16>
    return
  }
}
