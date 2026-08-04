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
} tpu_kernel_api_fill_kernel_t;

tpu_kernel_api_fill_kernel_t fill_fill_kernel_struct(unsigned long long out_v1);

int fill_kernel_check_mem(unsigned long long out_v1);


int fill_kernel_check_mem_s(tpu_kernel_api_fill_kernel_t *api);

int fill_kernel(tpudnnHandle_t t_handle, unsigned long long out_v1);

#ifdef __cplusplus
}
#endif
