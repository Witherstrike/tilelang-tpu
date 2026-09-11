#include <tpuv7_rt.h>
#include "kernel.h"
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>

tpuRtStream_t stream = nullptr;
tpuRtKernelModule_t tpu_module = nullptr;

static int checked(tpuRtStatus_t status, const char *operation) {{
  if (status != tpuRtSuccess) {{
    fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(status));
    return static_cast<int>(status);
  }}
  return 0;
}}

int init() {{
  const char *kernel_path = getenv("PPL_KERNEL_PATH");
  if (!kernel_path || !*kernel_path) return -2;
  int device = 0;
  if (const char *value = getenv("TILELANG_TPU_DEVICE_ID")) {{
    char *end = nullptr;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || end == value || *end || parsed < 0 || parsed > INT_MAX) {{
      fprintf(stderr, "Invalid TILELANG_TPU_DEVICE_ID: %s\n", value);
      return -2;
    }}
    device = static_cast<int>(parsed);
  }}
  int ret = checked(tpuRtInit(), "tpuRtInit");
  if (ret) return ret;
  ret = checked(tpuRtSetDevice(device), "tpuRtSetDevice");
  if (ret) return ret;
  ret = checked(tpuRtStreamCreate(&stream), "tpuRtStreamCreate");
  if (ret) return ret;
  tpu_module = tpuRtKernelLoadModuleFile(kernel_path, stream);
  if (!tpu_module) {{
    fprintf(stderr, "tpuRtKernelLoadModuleFile failed\n");
    checked(tpuRtStreamDestroy(stream), "tpuRtStreamDestroy");
    stream = nullptr;
    return -2;
  }}
  return 0;
}}

int post() {{
  int ret = checked(tpuRtKernelUnloadModule(tpu_module, stream), "tpuRtKernelUnloadModule");
  int destroy_ret = checked(tpuRtStreamDestroy(stream), "tpuRtStreamDestroy");
  tpu_module = nullptr;
  stream = nullptr;
  return ret ? ret : destroy_ret;
}}

extern "C" int tilelang_tpu_run(void** args) {{
{arg_declarations}
{device_declarations}
  int rst = init();
  if (rst) return rst;
  int cleanup_status = 0;
  int post_status = 0;
#define TPU_CHECK(call) do {{ rst = checked((call), #call); if (rst) goto cleanup; }} while (0)
{malloc_statements}
{memcpy_s2d_statements}
{kernel_call}
  if (rst) goto cleanup;
{memcpy_d2s_statements}
cleanup:
{free_statements}
  post_status = post();
#undef TPU_CHECK
  return rst ? rst : (cleanup_status ? cleanup_status : post_status);
}}
