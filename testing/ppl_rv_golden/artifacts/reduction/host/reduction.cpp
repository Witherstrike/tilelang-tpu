#include "reduction.h"
#include <ppl_mem.h>
#include <cstdio>
#include <mutex>
#include <memory>
#include <sstream>
#include <vector>
#include <unistd.h>
#include <numeric>
#include <cstring>
#include <string>

#define MIN(x, y) (((x)) < ((y)) ? (x) : (y))
#define MAX(x, y) (((x)) > ((y)) ? (x) : (y))

int reduction_kernel_check_mem(unsigned long long out_v1, unsigned long long in_v2) {
  return 0;
}
int reduction_kernel_check_mem_s(tpu_kernel_api_reduction_kernel_t *api) {
  return 0;
}

tpu_kernel_api_reduction_kernel_t fill_reduction_kernel_struct(unsigned long long out_v1, unsigned long long in_v2) {
  tpu_kernel_api_reduction_kernel_t api;
  api.out = out_v1;
  api.in = in_v2;
  return api;
}

int reduction_kernel(tpudnnHandle_t t_handle, unsigned long long out_v1, unsigned long long in_v2) {
  tpu_kernel_api_reduction_kernel_t api;
  int ret = -1;
  api.out = out_v1;
  api.in = in_v2;
  int group_num = 1;
  int block_num = 1;
  ret = tpudnnLaunchKernel(t_handle, "reduction_kernel_entry", &api,sizeof(api), group_num, block_num);
  if (ret != 0) {
      printf("tpu kernel launch failed!");
      return ret;
  }
  return 0;
}

