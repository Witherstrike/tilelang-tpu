#include "tpu_kernel.h"

#include "ppl_helper.h"

#include "atomic_def.h"

#include "rvt_api.h"

#include "assert.h"

static const int __alignTrans[] = {CONTINUOUS_LAYOUT, HW_ALIGN_LAYOUT, CONTINUOUS_LAYOUT, ROW_ALIGN_LAYOUT, FREE_LAYOUT};
typedef struct {
  global_addr_t out;
  global_addr_t in;
} tpu_kernel_api_copy_kernel_t;
void copy_kernel_inner(global_addr_t out_v1, global_addr_t in_v2) {
  rvt_cfg_lanemask(gdma_get_lane_mask());
  int32_t v3_32 = 32;
  int32_t v4_33 = 33;
  int32_t v5_8 = 8;
  int32_t v6_1 = 1;
  int32_t v7_0 = 0;
  int32_t v8_16 = 16;
  // copy.pl:6  ->
  int64_t v9 = (int64_t)in_v2; // <-
  // copy.pl:6  ->
  RVT_CFGGR(v3_32, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v9 /*addr*/); // <-
  // copy.pl:6  ->
  RVT_CFGGR_REG_SHAPE(v3_32 /*reg*/, v6_1 /*n*/, v5_8 /*c*/, v6_1 /*h*/, v8_16 /*w*/);
  ; // <-
  // copy.pl:7  ->
  int64_t v10 = (int64_t)out_v1; // <-
  // copy.pl:7  ->
  RVT_CFGGR(v4_33, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v10 /*addr*/); // <-
  // copy.pl:7  ->
  RVT_CFGGR_REG_SHAPE(v4_33 /*reg*/, v6_1 /*n*/, v5_8 /*c*/, v6_1 /*h*/, v8_16 /*w*/);
  ; // <-
  // copy.pl:8  ->
  RVT_CFGTR(v5_8, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, HW_ALIGN_LAYOUT /*layout*/, v7_0 /*addr*/); // <-
  // copy.pl:8  ->
  RVT_CFGTR_SHAPE(v5_8 /*reg*/, v6_1 /*n*/, v5_8 /*c*/, v6_1 /*h*/, v8_16 /*w*/);
  ; // <-
  // copy.pl:9  ->
  rvt_dma_ld(v5_8, v3_32); // <-
  // copy.pl:10  ->
  rvt_dma_st(v4_33, v5_8); // <-
  return;
}
int copy_kernel_entry(const void * args) {
  tpu_kernel_api_copy_kernel_t *api = (tpu_kernel_api_copy_kernel_t*)args;
  rvt_kernel_start();
  copy_kernel_inner(api->out,
    api->in);
  rvt_sync_i(0xdeadbeef, 0);
  return 0;
}
