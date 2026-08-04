#include "tpu_kernel.h"

#include "ppl_helper.h"

#include "atomic_def.h"

#include "rvt_api.h"

#include "assert.h"

static const int __alignTrans[] = {CONTINUOUS_LAYOUT, HW_ALIGN_LAYOUT, CONTINUOUS_LAYOUT, ROW_ALIGN_LAYOUT, FREE_LAYOUT};
typedef struct {
  global_addr_t out;
} tpu_kernel_api_fill_kernel_t;
void fill_kernel_inner(global_addr_t out_v1) {
  rvt_cfg_lanemask(gdma_get_lane_mask());
  int32_t v2_32 = 32;
  int32_t v3_8 = 8;
  int32_t v4_1 = 1;
  int32_t v5_0 = 0;
  float v6_1_500e0 = (float)1.500000000e+00;
  int32_t v7_16 = 16;
  // fill.pl:6  ->
  int64_t v8 = (int64_t)out_v1; // <-
  // fill.pl:6  ->
  RVT_CFGGR(v2_32, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v8 /*addr*/); // <-
  // fill.pl:6  ->
  RVT_CFGGR_REG_SHAPE(v2_32 /*reg*/, v4_1 /*n*/, v3_8 /*c*/, v4_1 /*h*/, v7_16 /*w*/);
  ; // <-
  // fill.pl:7  ->
  RVT_CFGTR(v3_8, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v5_0 /*addr*/); // <-
  // fill.pl:7  ->
  RVT_CFGTR_SHAPE(v3_8 /*reg*/, v4_1 /*n*/, v3_8 /*c*/, v4_1 /*h*/, v7_16 /*w*/);
  ; // <-
  // fill.pl:8  ->
  scalar_t v9_scalarT = {.u32 = 0};
  v9_scalarT.f32 = v6_1_500e0;
  v9_scalarT = tpu_cast(v9_scalarT, DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
  uint64_t v9 = v9_scalarT.u32;
  ; // <-
  // fill.pl:8  ->
  // dtype: DT_FP16
  int v10 = 1;
  RVT_CR(v10 /*reg*/, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, v9 /*val*/)
  ; // <-
  // fill.pl:8  ->
  rvt_cp(v3_8, v10); // <-
  // fill.pl:9  ->
  rvt_dma_st(v2_32, v3_8); // <-
  return;
}
int fill_kernel_entry(const void * args) {
  tpu_kernel_api_fill_kernel_t *api = (tpu_kernel_api_fill_kernel_t*)args;
  rvt_kernel_start();
  fill_kernel_inner(api->out);
  rvt_sync_i(0xdeadbeef, 0);
  return 0;
}
