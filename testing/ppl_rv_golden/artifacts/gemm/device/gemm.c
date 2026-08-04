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
} tpu_kernel_api_gemm_kernel_t;
void gemm_kernel_inner(global_addr_t out_v1, global_addr_t lhs_v2, global_addr_t rhs_v3) {
  rvt_cfg_lanemask(gdma_get_lane_mask());
  int32_t v4_33 = 33;
  int32_t v5_32 = 32;
  int32_t v6_34 = 34;
  int32_t v7_9 = 9;
  int32_t v8_10 = 10;
  int32_t v9_8 = 8;
  int32_t bias_v10_0 = 0;
  int32_t v11_1 = 1;
  int32_t v12_32768 = 32768;
  int32_t v13_16384 = 16384;
  bool v14_0 = false;
  int32_t v15_16 = 16;
  // gemm.pl:8  ->
  int64_t v16 = (int64_t)lhs_v2; // <-
  // gemm.pl:8  ->
  RVT_CFGGR(v4_33, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v16 /*addr*/); // <-
  // gemm.pl:8  ->
  RVT_CFGGR_REG_SHAPE(v4_33 /*reg*/, v11_1 /*n*/, v15_16 /*c*/, v11_1 /*h*/, v15_16 /*w*/);
  ; // <-
  // gemm.pl:9  ->
  int64_t v17 = (int64_t)rhs_v3; // <-
  // gemm.pl:9  ->
  RVT_CFGGR(v5_32, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v17 /*addr*/); // <-
  // gemm.pl:9  ->
  RVT_CFGGR_REG_SHAPE(v5_32 /*reg*/, v11_1 /*n*/, v15_16 /*c*/, v11_1 /*h*/, v15_16 /*w*/);
  ; // <-
  // gemm.pl:10  ->
  int64_t v18 = (int64_t)out_v1; // <-
  // gemm.pl:10  ->
  RVT_CFGGR(v6_34, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v18 /*addr*/); // <-
  // gemm.pl:10  ->
  RVT_CFGGR_REG_SHAPE(v6_34 /*reg*/, v11_1 /*n*/, v15_16 /*c*/, v11_1 /*h*/, v15_16 /*w*/);
  ; // <-
  // gemm.pl:11  ->
  RVT_CFGTR(v7_9, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v12_32768 /*addr*/); // <-
  // gemm.pl:11  ->
  RVT_CFGTR_SHAPE(v7_9 /*reg*/, v11_1 /*n*/, v15_16 /*c*/, v11_1 /*h*/, v15_16 /*w*/);
  ; // <-
  // gemm.pl:11  ->
  RVT_CFGTR(v8_10, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v13_16384 /*addr*/); // <-
  // gemm.pl:11  ->
  RVT_CFGTR_SHAPE(v8_10 /*reg*/, v11_1 /*n*/, v15_16 /*c*/, v11_1 /*h*/, v15_16 /*w*/);
  ; // <-
  // gemm.pl:11  ->
  RVT_CFGTR(v9_8, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, bias_v10_0 /*addr*/); // <-
  // gemm.pl:11  ->
  RVT_CFGTR_SHAPE(v9_8 /*reg*/, v11_1 /*n*/, v15_16 /*c*/, v11_1 /*h*/, v15_16 /*w*/);
  ; // <-
  // gemm.pl:12  ->
  rvt_dma_ld(v7_9, v4_33); // <-
  // gemm.pl:13  ->
  rvt_dma_ld(v8_10, v5_32); // <-
  // ppl_tiu_func.h:507  ->
  int v19 = 0;
  ; // <-
  // ppl_tiu_func.h:507  ->
  rvt_cfg_quant(v19);
  ; // <-
  // ppl_tiu_func.h:507  ->
  rvt_cfg_satu(v14_0 /*i*/, bias_v10_0 /*f*/);
  ; // <-
  // ppl_tiu_func.h:507  ->
  if (v14_0) {
    rvt_fmm2a_nn(v9_8, v7_9, v8_10, bias_v10_0 ? v19 : 0, bias_v10_0, v14_0);
  } else {
    rvt_fmm2_nn(v9_8, v7_9, v8_10, bias_v10_0 ? v19 : 0, bias_v10_0, v14_0);
  }
  ; // <-
  // gemm.pl:15  ->
  rvt_dma_st(v6_34, v9_8); // <-
  return;
}
int gemm_kernel_entry(const void * args) {
  tpu_kernel_api_gemm_kernel_t *api = (tpu_kernel_api_gemm_kernel_t*)args;
  rvt_kernel_start();
  gemm_kernel_inner(api->out,
    api->lhs,
    api->rhs);
  rvt_sync_i(0xdeadbeef, 0);
  return 0;
}
