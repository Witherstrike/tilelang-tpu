#include "tpu_kernel.h"

#include "ppl_helper.h"

#include "atomic_def.h"

#include "rvt_api.h"

#include "assert.h"

static const int __alignTrans[] = {CONTINUOUS_LAYOUT, HW_ALIGN_LAYOUT, CONTINUOUS_LAYOUT, ROW_ALIGN_LAYOUT, FREE_LAYOUT};
typedef struct {
  global_addr_t out;
  global_addr_t lhs;
  global_addr_t rhs;
} tpu_kernel_api_elementwise_kernel_t;
void elementwise_kernel_inner(global_addr_t out_v1, global_addr_t lhs_v2, global_addr_t rhs_v3) {
  rvt_cfg_lanemask(gdma_get_lane_mask());
  int32_t v4_32 = 32;
  int32_t v5_33 = 33;
  int32_t v6_34 = 34;
  int32_t v7_10 = 10;
  int32_t v8_8 = 8;
  int32_t v9_9 = 9;
  int32_t v10_11 = 11;
  int32_t v11_0 = 0;
  int32_t v12_1 = 1;
  int32_t v13_32768 = 32768;
  int32_t v14_16384 = 16384;
  float v15_5_000e_n1 = (float)5.000000000e-01;
  int32_t v16_16 = 16;
  // elementwise.pl:6  ->
  int64_t v17 = (int64_t)lhs_v2; // <-
  // elementwise.pl:6  ->
  RVT_CFGGR(v4_32, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v17 /*addr*/); // <-
  // elementwise.pl:6  ->
  RVT_CFGGR_REG_SHAPE(v4_32 /*reg*/, v12_1 /*n*/, v8_8 /*c*/, v12_1 /*h*/, v16_16 /*w*/);
  ; // <-
  // elementwise.pl:7  ->
  int64_t v18 = (int64_t)rhs_v3; // <-
  // elementwise.pl:7  ->
  RVT_CFGGR(v5_33, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v18 /*addr*/); // <-
  // elementwise.pl:7  ->
  RVT_CFGGR_REG_SHAPE(v5_33 /*reg*/, v12_1 /*n*/, v8_8 /*c*/, v12_1 /*h*/, v16_16 /*w*/);
  ; // <-
  // elementwise.pl:8  ->
  int64_t v19 = (int64_t)out_v1; // <-
  // elementwise.pl:8  ->
  RVT_CFGGR(v6_34, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v19 /*addr*/); // <-
  // elementwise.pl:8  ->
  RVT_CFGGR_REG_SHAPE(v6_34 /*reg*/, v12_1 /*n*/, v8_8 /*c*/, v12_1 /*h*/, v16_16 /*w*/);
  ; // <-
  // elementwise.pl:9  ->
  RVT_CFGTR(v7_10, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v13_32768 /*addr*/); // <-
  // elementwise.pl:9  ->
  RVT_CFGTR_SHAPE(v7_10 /*reg*/, v12_1 /*n*/, v8_8 /*c*/, v12_1 /*h*/, v16_16 /*w*/);
  ; // <-
  // elementwise.pl:9  ->
  RVT_CFGTR(v8_8, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v11_0 /*addr*/); // <-
  // elementwise.pl:9  ->
  RVT_CFGTR_SHAPE(v8_8 /*reg*/, v12_1 /*n*/, v8_8 /*c*/, v12_1 /*h*/, v16_16 /*w*/);
  ; // <-
  // elementwise.pl:9  ->
  RVT_CFGTR(v9_9, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v14_16384 /*addr*/); // <-
  // elementwise.pl:9  ->
  RVT_CFGTR_SHAPE(v9_9 /*reg*/, v12_1 /*n*/, v8_8 /*c*/, v12_1 /*h*/, v16_16 /*w*/);
  ; // <-
  // elementwise.pl:9  ->
  RVT_CFGTR(v10_11, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v11_0 /*addr*/); // <-
  // elementwise.pl:9  ->
  RVT_CFGTR_SHAPE(v10_11 /*reg*/, v12_1 /*n*/, v8_8 /*c*/, v12_1 /*h*/, v16_16 /*w*/);
  ; // <-
  // elementwise.pl:10  ->
  rvt_dma_ld(v7_10, v4_32); // <-
  // elementwise.pl:11  ->
  rvt_dma_ld(v8_8, v5_33); // <-
  // elementwise.pl:12  ->
  rvt_cfg_satu(v11_0 /*i*/, v11_0 /*f*/);
  ; // <-
  // elementwise.pl:12  ->
  rvt_cfg_round_mode(v11_0);
  ; // <-
  // elementwise.pl:12  ->
  rvt_fadd(v9_9, v7_10, v8_8); // <-
  // elementwise.pl:13  ->
  scalar_t v20_scalarT = {.u32 = 0};
  v20_scalarT.f32 = v15_5_000e_n1;
  v20_scalarT = tpu_cast(v20_scalarT, DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
  uint64_t v20 = v20_scalarT.u32;
  ; // <-
  // elementwise.pl:13  ->
  // dtype: DT_FP16
  int v21 = 1;
  RVT_CR(v21 /*reg*/, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, v20 /*val*/)
  ; // <-
  // elementwise.pl:13  ->
  rvt_fmul(v10_11, v9_9, v21); // <-
  // elementwise.pl:14  ->
  rvt_dma_st(v6_34, v10_11); // <-
  return;
}
int elementwise_kernel_entry(const void * args) {
  tpu_kernel_api_elementwise_kernel_t *api = (tpu_kernel_api_elementwise_kernel_t*)args;
  rvt_kernel_start();
  elementwise_kernel_inner(api->out,
    api->lhs,
    api->rhs);
  rvt_sync_i(0xdeadbeef, 0);
  return 0;
}
