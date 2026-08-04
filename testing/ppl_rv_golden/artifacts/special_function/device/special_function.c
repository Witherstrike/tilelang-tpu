#include "tpu_kernel.h"

#include "ppl_helper.h"

#include "atomic_def.h"

#include "rvt_api.h"

#include "assert.h"

static const int __alignTrans[] = {CONTINUOUS_LAYOUT, HW_ALIGN_LAYOUT, CONTINUOUS_LAYOUT, ROW_ALIGN_LAYOUT, FREE_LAYOUT};
typedef struct {
  global_addr_t out;
  global_addr_t in;
} tpu_kernel_api_special_function_kernel_t;
void special_function_kernel_inner(global_addr_t out_v1, global_addr_t in_v2) {
  rvt_cfg_lanemask(gdma_get_lane_mask());
  int32_t v3_32 = 32;
  int32_t v4_33 = 33;
  int32_t v5_9 = 9;
  int32_t v6_8 = 8;
  int32_t v7_1 = 1;
  int32_t v8_16384 = 16384;
  int32_t v9_0 = 0;
  int32_t v10_3 = 3;
  int32_t v11_16 = 16;
  // special_function.pl:6  ->
  int64_t v12 = (int64_t)in_v2; // <-
  // special_function.pl:6  ->
  RVT_CFGGR(v3_32, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v12 /*addr*/); // <-
  // special_function.pl:6  ->
  RVT_CFGGR_REG_SHAPE(v3_32 /*reg*/, v7_1 /*n*/, v6_8 /*c*/, v7_1 /*h*/, v11_16 /*w*/);
  ; // <-
  // special_function.pl:7  ->
  int64_t v13 = (int64_t)out_v1; // <-
  // special_function.pl:7  ->
  RVT_CFGGR(v4_33, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v13 /*addr*/); // <-
  // special_function.pl:7  ->
  RVT_CFGGR_REG_SHAPE(v4_33 /*reg*/, v7_1 /*n*/, v6_8 /*c*/, v7_1 /*h*/, v11_16 /*w*/);
  ; // <-
  // special_function.pl:8  ->
  RVT_CFGTR(v5_9, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v8_16384 /*addr*/); // <-
  // special_function.pl:8  ->
  RVT_CFGTR_SHAPE(v5_9 /*reg*/, v7_1 /*n*/, v6_8 /*c*/, v7_1 /*h*/, v11_16 /*w*/);
  ; // <-
  // special_function.pl:8  ->
  RVT_CFGTR(v6_8, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v9_0 /*addr*/); // <-
  // special_function.pl:8  ->
  RVT_CFGTR_SHAPE(v6_8 /*reg*/, v7_1 /*n*/, v6_8 /*c*/, v7_1 /*h*/, v11_16 /*w*/);
  ; // <-
  // special_function.pl:9  ->
  rvt_dma_ld(v5_9, v3_32); // <-
  // special_function.pl:10  ->
  rvt_cfg_rsqrt_iter(v10_3);
  ; // <-
  // special_function.pl:10  ->
  rvt_sfu_rsqrt(v6_8, v5_9); // <-
  // special_function.pl:11  ->
  rvt_dma_st(v4_33, v6_8); // <-
  return;
}
int special_function_kernel_entry(const void * args) {
  tpu_kernel_api_special_function_kernel_t *api = (tpu_kernel_api_special_function_kernel_t*)args;
  rvt_kernel_start();
  special_function_kernel_inner(api->out,
    api->in);
  rvt_sync_i(0xdeadbeef, 0);
  return 0;
}
