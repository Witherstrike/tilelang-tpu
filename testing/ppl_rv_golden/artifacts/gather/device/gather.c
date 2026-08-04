#include "tpu_kernel.h"

#include "ppl_helper.h"

#include "atomic_def.h"

#include "rvt_api.h"

#include "assert.h"

static const int __alignTrans[] = {CONTINUOUS_LAYOUT, HW_ALIGN_LAYOUT, CONTINUOUS_LAYOUT, ROW_ALIGN_LAYOUT, FREE_LAYOUT};
typedef struct {
  global_addr_t out;
  global_addr_t table;
  global_addr_t index;
} tpu_kernel_api_gather_kernel_t;
void gather_kernel_inner(global_addr_t out_v1, global_addr_t table_v2, global_addr_t index_v3) {
  rvt_cfg_lanemask(gdma_get_lane_mask());
  int32_t v4_33 = 33;
  int32_t v5_32 = 32;
  int32_t v6_34 = 34;
  int32_t v7_0 = 0;
  int32_t v8_1 = 1;
  int32_t v9_4 = 4;
  int32_t v10_16 = 16;
  // gather.pl:8  ->
  int64_t v11 = (int64_t)table_v2; // <-
  // gather.pl:8  ->
  RVT_CFGGR(v4_33, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v11 /*addr*/); // <-
  // gather.pl:8  ->
  RVT_CFGGR_REG_SHAPE(v4_33 /*reg*/, v8_1 /*n*/, v8_1 /*c*/, v10_16 /*h*/, v10_16 /*w*/);
  ; // <-
  // gather.pl:9  ->
  int64_t v12 = (int64_t)index_v3; // <-
  // gather.pl:9  ->
  RVT_CFGGR(v5_32, 0 /*is_int*/, TEEW_E32 /*DT_UINT32*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v12 /*addr*/); // <-
  // gather.pl:9  ->
  RVT_CFGGR_REG_SHAPE(v5_32 /*reg*/, v8_1 /*n*/, v8_1 /*c*/, v9_4 /*h*/, v8_1 /*w*/);
  ; // <-
  // gather.pl:10  ->
  int64_t v13 = (int64_t)out_v1; // <-
  // gather.pl:10  ->
  RVT_CFGGR(v6_34, 0 /*is_int*/, TEEW_E16 /*DT_FP16*/, 0 /*subtype*/, CONTINUOUS_LAYOUT /*layout*/, v13 /*addr*/); // <-
  // gather.pl:10  ->
  RVT_CFGGR_REG_SHAPE(v6_34 /*reg*/, v8_1 /*n*/, v8_1 /*c*/, v9_4 /*h*/, v10_16 /*w*/);
  ; // <-
  // ppl_dma_func.h:284  ->
  scalar_t v14_scalarT = {.u32 = 0};
  v14_scalarT.s32 = v7_0;
  v14_scalarT = tpu_cast(v14_scalarT, DT_FP16, DT_INT32, RM_HALF_TO_EVEN);
  uint64_t v14 = v14_scalarT.u32;
  ; // <-
  // ppl_dma_func.h:284  ->
  rvt_cfg_dmaidx(v7_0, &v14); // <-
  // ppl_dma_func.h:284  ->
  rvt_dma_hgather(v6_34, v4_33, v5_32, v7_0); // <-
  return;
}
int gather_kernel_entry(const void * args) {
  tpu_kernel_api_gather_kernel_t *api = (tpu_kernel_api_gather_kernel_t*)args;
  rvt_kernel_start();
  gather_kernel_inner(api->out,
    api->table,
    api->index);
  rvt_sync_i(0xdeadbeef, 0);
  return 0;
}
