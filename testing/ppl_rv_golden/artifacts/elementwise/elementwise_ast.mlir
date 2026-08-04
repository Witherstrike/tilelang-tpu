module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @_ZN3ppl13set_block_numEi(i32 {ppl.vname = "num"}) attributes {ori_name = "set_block_num"}
  func.func @_ZN3ppl14get_ccl_msg_idEv() -> i32 attributes {llvm.linkage = #llvm.linkage<external>, ori_name = "get_ccl_msg_id"} {
    %c0_i32 = arith.constant 0 : i32
    %0 = call @_ZN3ppl10get_msg_idE13msg_id_type_t(%c0_i32) {ori_name = "get_msg_id"} : (i32) -> i32
    return %0 : i32
  }
  func.func private @_ZN3ppl10get_msg_idE13msg_id_type_t(i32 {ppl.vname = "type"}) -> i32 attributes {ori_name = "get_msg_id", unsignedIdx = [0 : index]}
  func.func private @_ZN3ppl11sync_engineEibi(i32 {ppl.vname = "engines_type"}, i1 {ppl.vname = "all_core"}, i32 {ppl.vname = "block_num"}) attributes {ori_name = "sync_engine", unsignedIdx = [1 : index]}
  func.func @_ZN3ppl8sync_allEi(%arg0: i32 {ppl.vname = "block_num"}) attributes {llvm.linkage = #llvm.linkage<external>, ori_name = "sync_all"} {
    %true = arith.constant true
    %c15_i32 = arith.constant 15 : i32
    call @_ZN3ppl11sync_engineEibi(%c15_i32, %true, %arg0) {ori_name = "sync_engine"} : (i32, i1, i32) -> ()
    return
  }
  func.func private @_ZN3ppl13send_msg_baseEiii13engine_type_t(i32 {ppl.vname = "msg_id"}, i32 {ppl.vname = "send_cnt"}, i32 {ppl.vname = "port"}, i32 {ppl.vname = "engine"}) -> i32 attributes {ori_name = "send_msg_base", unsignedIdx = [3 : index]}
  func.func private @_ZN3ppl13wait_msg_baseEiii13engine_type_t(i32 {ppl.vname = "msg_id"}, i32 {ppl.vname = "wait_cnt"}, i32 {ppl.vname = "port"}, i32 {ppl.vname = "engine"}) -> i32 attributes {ori_name = "wait_msg_base", unsignedIdx = [3 : index]}
  func.func private @_ZN3ppl30set_config_memory_alloc_methodE21memory_alloc_method_t(i32 {ppl.vname = "method"}) attributes {ori_name = "set_config_memory_alloc_method", unsignedIdx = [0 : index]}
  func.func private @_ZN3ppl4dim4C1Eiiii(memref<?x4xi32> {ppl.vname = "_n"}, i32 {ppl.vname = "_c"}, i32 {ppl.vname = "_h"}, i32 {ppl.vname = "_w"}, i32) attributes {ori_name = "dim4"}
  func.func @_ZN3ppl5vsdma8send_msgEiii(%arg0: i32 {ppl.vname = "msg_id"}, %arg1: i32 {ppl.vname = "msg_cnt"}, %arg2: i32 {ppl.vname = "port_id"}) attributes {llvm.linkage = #llvm.linkage<external>, ori_name = "send_msg"} {
    %c16_i32 = arith.constant 16 : i32
    %0 = call @_ZN3ppl13send_msg_baseEiii13engine_type_t(%arg0, %arg1, %arg2, %c16_i32) {ori_name = "send_msg_base"} : (i32, i32, i32, i32) -> i32
    return
  }
  func.func @_ZN3ppl5vsdma8wait_msgEiii(%arg0: i32 {ppl.vname = "msg_id"}, %arg1: i32 {ppl.vname = "msg_cnt"}, %arg2: i32 {ppl.vname = "port_id"}) attributes {llvm.linkage = #llvm.linkage<external>, ori_name = "wait_msg"} {
    %c16_i32 = arith.constant 16 : i32
    %0 = call @_ZN3ppl13wait_msg_baseEiii13engine_type_t(%arg0, %arg1, %arg2, %c16_i32) {ori_name = "wait_msg_base"} : (i32, i32, i32, i32) -> i32
    return
  }
  func.func private @_ZN3ppl4cdma3nopEi(i32 {ppl.vname = "port"}) attributes {ori_name = "nop"}
  func.func private @_ZN3ppl4cdma11tx_send_msgEiii(i32 {ppl.vname = "port"}, i32 {ppl.vname = "msg_id"}, i32 {ppl.vname = "wait_cnt"}) attributes {ori_name = "tx_send_msg"}
  func.func private @_ZN3ppl4cdma11tx_wait_msgEiii(i32 {ppl.vname = "port"}, i32 {ppl.vname = "msg_id"}, i32 {ppl.vname = "send_cnt"}) attributes {ori_name = "tx_wait_msg"}
  func.func @_Z18elementwise_kernelP4fp16S0_S0_(%arg0: memref<?xf16> {ppl.vname = "out"}, %arg1: memref<?xf16> {ppl.vname = "lhs"}, %arg2: memref<?xf16> {ppl.vname = "rhs"}) attributes {category = "kernel", ori_name = "elementwise_kernel"} {
    %false = arith.constant false
    %cst = arith.constant 5.000000e-01 : f32
    %c-1_i64 = arith.constant -1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() {ppl.vname = "result"} : memref<1x1xmemref<?xf16>>
    %cast = memref.cast %alloca : memref<1x1xmemref<?xf16>> to memref<?x1xmemref<?xf16>>
    %alloca_0 = memref.alloca() {ppl.vname = "sum"} : memref<1x1xmemref<?xf16>>
    %cast_1 = memref.cast %alloca_0 : memref<1x1xmemref<?xf16>> to memref<?x1xmemref<?xf16>>
    %alloca_2 = memref.alloca() {ppl.vname = "r"} : memref<1x1xmemref<?xf16>>
    %cast_3 = memref.cast %alloca_2 : memref<1x1xmemref<?xf16>> to memref<?x1xmemref<?xf16>>
    %alloca_4 = memref.alloca() {ppl.vname = "l"} : memref<1x1xmemref<?xf16>>
    %cast_5 = memref.cast %alloca_4 : memref<1x1xmemref<?xf16>> to memref<?x1xmemref<?xf16>>
    %alloca_6 = memref.alloca() {ppl.vname = "g_out"} : memref<1x1xmemref<?xf16>>
    %cast_7 = memref.cast %alloca_6 : memref<1x1xmemref<?xf16>> to memref<?x1xmemref<?xf16>>
    %alloca_8 = memref.alloca() {ppl.vname = "g_rhs"} : memref<1x1xmemref<?xf16>>
    %cast_9 = memref.cast %alloca_8 : memref<1x1xmemref<?xf16>> to memref<?x1xmemref<?xf16>>
    %alloca_10 = memref.alloca() {ppl.vname = "g_lhs"} : memref<1x1xmemref<?xf16>>
    %cast_11 = memref.cast %alloca_10 : memref<1x1xmemref<?xf16>> to memref<?x1xmemref<?xf16>>
    %alloca_12 = memref.alloca() {ppl.vname = "shape"} : memref<1x4xi32>
    %cast_13 = memref.cast %alloca_12 : memref<1x4xi32> to memref<?x4xi32>
    call @_ZN3ppl4dim4C1Eiiii(%cast_13, %c1_i32, %c8_i32, %c1_i32, %c16_i32) {ori_name = "dim4"} : (memref<?x4xi32>, i32, i32, i32, i32) -> ()
    call @_ZN3ppl7gtensorI4fp16EC1IRNS_4dim4EEEOT_13tensor_mode_tPS1_(%cast_11, %cast_13, %c2_i32, %arg1) {ori_name = "gtensor"} : (memref<?x1xmemref<?xf16>>, memref<?x4xi32>, i32, memref<?xf16>) -> ()
    call @_ZN3ppl7gtensorI4fp16EC1IRNS_4dim4EEEOT_13tensor_mode_tPS1_(%cast_9, %cast_13, %c2_i32, %arg2) {ori_name = "gtensor"} : (memref<?x1xmemref<?xf16>>, memref<?x4xi32>, i32, memref<?xf16>) -> ()
    call @_ZN3ppl7gtensorI4fp16EC1IRNS_4dim4EEEOT_13tensor_mode_tPS1_(%cast_7, %cast_13, %c2_i32, %arg0) {ori_name = "gtensor"} : (memref<?x1xmemref<?xf16>>, memref<?x4xi32>, i32, memref<?xf16>) -> ()
    call @_ZN3ppl6tensorI4fp16EC1IRNS_4dim4EEEOT_12align_mode_tx(%cast_5, %cast_13, %c1_i32, %c-1_i64) {ori_name = "tensor"} : (memref<?x1xmemref<?xf16>>, memref<?x4xi32>, i32, i64) -> ()
    call @_ZN3ppl6tensorI4fp16EC1IRNS_4dim4EEEOT_12align_mode_tx(%cast_3, %cast_13, %c1_i32, %c-1_i64) {ori_name = "tensor"} : (memref<?x1xmemref<?xf16>>, memref<?x4xi32>, i32, i64) -> ()
    call @_ZN3ppl6tensorI4fp16EC1IRNS_4dim4EEEOT_12align_mode_tx(%cast_1, %cast_13, %c1_i32, %c-1_i64) {ori_name = "tensor"} : (memref<?x1xmemref<?xf16>>, memref<?x4xi32>, i32, i64) -> ()
    call @_ZN3ppl6tensorI4fp16EC1IRNS_4dim4EEEOT_12align_mode_tx(%cast, %cast_13, %c1_i32, %c-1_i64) {ori_name = "tensor"} : (memref<?x1xmemref<?xf16>>, memref<?x4xi32>, i32, i64) -> ()
    call @_ZN3ppl3dma4loadI4fp16EEvRNS_6tensorIT_EERNS_7gtensorIS4_EE(%cast_5, %cast_11) {ori_name = "load"} : (memref<?x1xmemref<?xf16>>, memref<?x1xmemref<?xf16>>) -> ()
    call @_ZN3ppl3dma4loadI4fp16EEvRNS_6tensorIT_EERNS_7gtensorIS4_EE(%cast_3, %cast_9) {ori_name = "load"} : (memref<?x1xmemref<?xf16>>, memref<?x1xmemref<?xf16>>) -> ()
    call @_ZN3ppl3tiu4faddI4fp16EEvRNS_6tensorIT_EES6_S6_b(%cast_1, %cast_5, %cast_3, %false) {ori_name = "fadd"} : (memref<?x1xmemref<?xf16>>, memref<?x1xmemref<?xf16>>, memref<?x1xmemref<?xf16>>, i1) -> ()
    call @_ZN3ppl3tiu4fmulI4fp16EEvRNS_6tensorIT_EES6_fb(%cast, %cast_1, %cst, %false) {ori_name = "fmul"} : (memref<?x1xmemref<?xf16>>, memref<?x1xmemref<?xf16>>, f32, i1) -> ()
    call @_ZN3ppl3dma5storeI4fp16EEvRNS_7gtensorIT_EERNS_6tensorIS4_EE(%cast_7, %cast) {ori_name = "store"} : (memref<?x1xmemref<?xf16>>, memref<?x1xmemref<?xf16>>) -> ()
    return
  }
  func.func private @_ZN3ppl7gtensorI4fp16EC1IRNS_4dim4EEEOT_13tensor_mode_tPS1_(memref<?x1xmemref<?xf16>> {ppl.vname = "_shape"}, memref<?x4xi32> {ppl.vname = "mode"}, i32 {ppl.vname = "address"}, memref<?xf16>) attributes {ori_name = "gtensor", unsignedIdx = [1 : index]}
  func.func private @_ZN3ppl6tensorI4fp16EC1IRNS_4dim4EEEOT_12align_mode_tx(memref<?x1xmemref<?xf16>> {ppl.vname = "_shape"}, memref<?x4xi32> {ppl.vname = "align_mode"}, i32 {ppl.vname = "address"}, i64) attributes {ori_name = "tensor", unsignedIdx = [1 : index]}
  func.func private @_ZN3ppl3dma4loadI4fp16EEvRNS_6tensorIT_EERNS_7gtensorIS4_EE(memref<?x1xmemref<?xf16>> {ppl.vname = "dst"}, memref<?x1xmemref<?xf16>> {ppl.vname = "src"}) attributes {ori_name = "load"}
  func.func private @_ZN3ppl3tiu4faddI4fp16EEvRNS_6tensorIT_EES6_S6_b(memref<?x1xmemref<?xf16>> {ppl.vname = "dst"}, memref<?x1xmemref<?xf16>> {ppl.vname = "src0"}, memref<?x1xmemref<?xf16>> {ppl.vname = "src1"}, i1 {ppl.vname = "saturation"}) attributes {ori_name = "fadd", unsignedIdx = [3 : index]}
  func.func private @_ZN3ppl3tiu4fmulI4fp16EEvRNS_6tensorIT_EES6_fb(memref<?x1xmemref<?xf16>> {ppl.vname = "dst"}, memref<?x1xmemref<?xf16>> {ppl.vname = "src"}, f32 {ppl.vname = "C"}, i1 {ppl.vname = "saturation"}) attributes {ori_name = "fmul", unsignedIdx = [3 : index]}
  func.func private @_ZN3ppl3dma5storeI4fp16EEvRNS_7gtensorIT_EERNS_6tensorIS4_EE(memref<?x1xmemref<?xf16>> {ppl.vname = "dst"}, memref<?x1xmemref<?xf16>> {ppl.vname = "src"}) attributes {ori_name = "store"}
}
