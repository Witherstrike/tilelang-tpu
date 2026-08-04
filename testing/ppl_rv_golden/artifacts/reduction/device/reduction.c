#include "tpu_kernel.h"

#include "ppl_helper.h"

#include "atomic_def.h"

#include "rvt_api.h"

#include "assert.h"

static const int __alignTrans[] = {CONTINUOUS_LAYOUT, HW_ALIGN_LAYOUT, CONTINUOUS_LAYOUT, ROW_ALIGN_LAYOUT, FREE_LAYOUT};
typedef struct {
  global_addr_t out;
  global_addr_t in;
} tpu_kernel_api_reduction_kernel_t;
void reduction_kernel_inner(global_addr_t out_v1, global_addr_t in_v2) {
  rvt_cfg_lanemask(gdma_get_lane_mask());
  int32_t v3_32 = 32;
  int32_t v4_33 = 33;
  int32_t v5_8 = 8;
  int32_t v6_9 = 9;
  int32_t v7_1 = 1;
  int32_t v8_16384 = 16384;
  int32_t v9_0 = 0;
  bool v10_0 = false;
  int32_t v11_16 = 16;
  // reduction.pl:7  ->
  int64_t v12 = (int64_t)in_v2; // <-
  // reduction.pl:7  ->
  RVT_CFGGR(v3_32, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v12 /*addr*/); // <-
  // reduction.pl:7  ->
  RVT_CFGGR_REG_SHAPE(v3_32 /*reg*/, v7_1 /*n*/, v5_8 /*c*/, v7_1 /*h*/, v11_16 /*w*/);
  ; // <-
  // reduction.pl:8  ->
  int64_t v13 = (int64_t)out_v1; // <-
  // reduction.pl:8  ->
  RVT_CFGGR(v4_33, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v13 /*addr*/); // <-
  // reduction.pl:8  ->
  RVT_CFGGR_REG_SHAPE(v4_33 /*reg*/, v7_1 /*n*/, v5_8 /*c*/, v7_1 /*h*/, v7_1 /*w*/);
  ; // <-
  // reduction.pl:9  ->
  RVT_CFGTR(v5_8, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v8_16384 /*addr*/); // <-
  // reduction.pl:9  ->
  RVT_CFGTR_SHAPE(v5_8 /*reg*/, v7_1 /*n*/, v5_8 /*c*/, v7_1 /*h*/, v11_16 /*w*/);
  ; // <-
  // reduction.pl:9  ->
  RVT_CFGTR(v6_9, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v9_0 /*addr*/); // <-
  // reduction.pl:9  ->
  RVT_CFGTR_SHAPE(v6_9 /*reg*/, v7_1 /*n*/, v5_8 /*c*/, v7_1 /*h*/, v7_1 /*w*/);
  ; // <-
  // reduction.pl:10  ->
  rvt_dma_ld(v5_8, v3_32); // <-
  // reduction.pl:11  ->
  rvt_cfg_satu(v10_0 /*i*/, v10_0 /*f*/);
  ; // <-
  // reduction.pl:11  ->
  TPUKERNEL_ASSERT( 0 && "Chip have no instruction like TiuReduceOp!!!\n");
  ; // <-
  // reduction.pl:12  ->
  rvt_dma_st(v4_33, v6_9); // <-
  return;
}
int reduction_kernel_entry(const void * args) {
  tpu_kernel_api_reduction_kernel_t *api = (tpu_kernel_api_reduction_kernel_t*)args;
  rvt_kernel_start();
  reduction_kernel_inner(api->out,
    api->in);
  rvt_sync_i(0xdeadbeef, 0);
  return 0;
}
