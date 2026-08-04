#pragma once
#ifndef __tpub_7_1_e_rv__
#define __tpub_7_1_e_rv__
#endif

#include <stdint.h>
#include <assert.h>
#include "tpuv7_rt.h"
#include <tpuDNN.h>

#include "host_utils.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long long out;
  unsigned long long lhs;
  unsigned long long rhs;
} tpu_kernel_api_gemm_kernel_t;

tpu_kernel_api_gemm_kernel_t fill_gemm_kernel_struct(unsigned long long out_v1, unsigned long long lhs_v2, unsigned long long rhs_v3);

int gemm_kernel_check_mem(unsigned long long out_v1, unsigned long long lhs_v2, unsigned long long rhs_v3);


int gemm_kernel_check_mem_s(tpu_kernel_api_gemm_kernel_t *api);

int gemm_kernel(tpudnnHandle_t t_handle, unsigned long long out_v1, unsigned long long lhs_v2, unsigned long long rhs_v3);

#ifdef __cplusplus
}
#endif
