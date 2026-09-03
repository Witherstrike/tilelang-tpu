#include <tpuv7_rt.h>
#include "kernel.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

tpuRtStream_t stream = nullptr;
tpuRtKernelModule_t tpu_module = nullptr;
static std::mutex tilelang_tpu_profile_mutex;
static int tilelang_tpu_expected_device_id = -1;

// LibraryGenerator calls this immediately after dlopen, before tilelang_tpu_run
// can initialize the vendor runtime.  Keep the expected device inside main.so
// as well as Python's process-wide profile: a caller changing the environment
// between dlopen and dispatch must fail before tpuRtInit/tpuRtSetDevice.
extern "C" int tilelang_tpu_bind_device(int expected_device_id) {{
  if (expected_device_id < 0) {{
    return -1;
  }}
#ifdef USING_CMODEL
  if (expected_device_id != 0) {{
    return -2;
  }}
#endif
  std::lock_guard<std::mutex> lock(tilelang_tpu_profile_mutex);
  if (tilelang_tpu_expected_device_id >= 0 &&
      tilelang_tpu_expected_device_id != expected_device_id) {{
    return -3;
  }}
  tilelang_tpu_expected_device_id = expected_device_id;
  return 0;
}}

static int tilelang_tpu_device_id() {{
#ifdef USING_CMODEL
  return 0;
#else
  // Board execution is deliberately fail-closed.  The Python loader blocks
  // PCIe dlopen too; retain the same gate here so a manually loaded main.so
  // cannot reach tpuRtInit without an explicit acknowledgement.
  const char* allow_pcie = std::getenv("TILELANG_TPU_ALLOW_PCIE_LOAD");
  if (allow_pcie == nullptr || std::strcmp(allow_pcie, "1") != 0) {{
    std::cerr << "Set TILELANG_TPU_ALLOW_PCIE_LOAD=1 before a PCIe TPU dispatch.\n";
    return -1;
  }}
  // A caller must also name the intended PCIe device rather than inheriting
  // the historical hard-coded device ID 14.
  const char* value = std::getenv("TILELANG_TPU_DEVICE_ID");
  if (value == nullptr || *value == '\0') {{
    std::cerr << "Set TILELANG_TPU_DEVICE_ID before a PCIe TPU dispatch.\n";
    return -1;
  }}
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {{
    std::cerr << "Invalid TILELANG_TPU_DEVICE_ID: " << value << "\n";
    return -1;
  }}
  return static_cast<int>(parsed);
#endif
}}

static const char* tilelang_tpu_kernel_path() {{
#ifdef TILELANG_PPL_KERNEL_PATH
  return TILELANG_PPL_KERNEL_PATH;
#else
  // Keep the standalone template usable, while JIT-built main.so embeds its
  // private libkernel.so path and never relies on this process-global value.
  return std::getenv("PPL_KERNEL_PATH");
#endif
}}

int init() {{
#ifdef USING_CMODEL
#ifndef TILELANG_TPU_CMODEL_CORE_NUM
#error "CModel main.so must embed the target core count"
#endif
  // This must happen in the process that calls tpuRtInit.  Setting it only in
  // the compiler process breaks a fresh cached/from-database CModel process.
  if (setenv("TPU_RT_CORE_NUM", TILELANG_TPU_CMODEL_CORE_NUM, 1) != 0) {{
    return -9;
  }}
#endif
  const int device_id = tilelang_tpu_device_id();
  if (device_id < 0) {{
    return -2;
  }}
  {{
    std::lock_guard<std::mutex> lock(tilelang_tpu_profile_mutex);
    if (tilelang_tpu_expected_device_id < 0 ||
        tilelang_tpu_expected_device_id != device_id) {{
      // Do not allow an environment change after LibraryGenerator.load_lib()
      // to choose another board (or make a CModel library look like PCIe).
      return -10;
    }}
  }}
  tpuRtStatus_t ret = tpuRtInit();
  if (ret != tpuRtSuccess) {{
    return -1;
  }}
  ret = tpuRtSetDevice(device_id);
  if (ret != tpuRtSuccess) {{
    return -3;
  }}
  ret = tpuRtStreamCreate(&stream);
  if (ret != tpuRtSuccess) {{
    stream = nullptr;
    return -4;
  }}
  const char* kernel_dir = tilelang_tpu_kernel_path();
  if (kernel_dir == nullptr || *kernel_dir == '\0') {{
    tpuRtStreamDestroy(stream);
    stream = nullptr;
    return -5;
  }}
  tpu_module = tpuRtKernelLoadModuleFile(kernel_dir, stream);
  if (tpu_module == nullptr) {{
    tpuRtStreamDestroy(stream);
    stream = nullptr;
    return -6;
  }}
  return 0;
}}

void post() {{
  if (tpu_module != nullptr) {{
    tpuRtKernelUnloadModule(tpu_module, stream);
    tpu_module = nullptr;
  }}
  if (stream != nullptr) {{
    tpuRtStreamDestroy(stream);
    stream = nullptr;
  }}
}}

extern "C" int tilelang_tpu_run(void** args) {{
  if (args == nullptr) {{
    return -7;
  }}
{arg_declarations}

  int status = init();
  if (status != 0) {{
    return status;
  }}

  // Device pointers are initialized so cleanup remains safe after a partial
  // allocation or transfer failure.
{device_declarations}

  do {{
{malloc_statements}
    if (status != 0) {{
      break;
    }}
{memcpy_s2d_statements}
    if (status != 0) {{
      break;
    }}

    auto start = std::chrono::high_resolution_clock::now();
{kernel_call}
    auto end = std::chrono::high_resolution_clock::now();
    if (rst != 0) {{
      std::cerr << "kernel_launch failed: " << rst << "\n";
      status = rst;
      break;
    }}
    std::cout << "kernel_launch success\n";

    const auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    const double elapsed_time_ms = duration.count() / 1000.0;
    std::cout << "Single kernel execution time: " << elapsed_time_ms << " ms ("
              << duration.count() << " us)\n";

    // Benchmarking is opt-in. Its default is zero extra launches for both
    // CModel and PCIe, so a first PCIe smoke remains exactly one dispatch.
    int measure_runs = 0;
    if (const char* value = std::getenv("TILELANG_TPU_BENCHMARK_RUNS")) {{
      measure_runs = std::max(0, std::atoi(value));
    }}
    if (measure_runs > 0) {{
      const int warmup_runs = std::min(5, measure_runs);
      std::cout << "\n=== Performance Benchmark (after " << warmup_runs
                << " warmup runs) ===\n";
      for (int i = 0; i < warmup_runs; ++i) {{
{pure_kernel_call}
        if (rst != 0) {{
          status = rst;
          break;
        }}
      }}
      if (status != 0) {{
        break;
      }}
      double total_time_us = 0.0;
      double min_time_us = std::numeric_limits<double>::max();
      double max_time_us = 0.0;
      for (int i = 0; i < measure_runs; ++i) {{
        const auto run_start = std::chrono::high_resolution_clock::now();
{pure_kernel_call}
        const auto run_end = std::chrono::high_resolution_clock::now();
        if (rst != 0) {{
          status = rst;
          break;
        }}
        const double run_time_us =
            std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start)
                .count();
        total_time_us += run_time_us;
        min_time_us = std::min(min_time_us, run_time_us);
        max_time_us = std::max(max_time_us, run_time_us);
      }}
      if (status != 0) {{
        break;
      }}
      std::cout << "Runs: " << measure_runs << ", average: "
                << total_time_us / measure_runs / 1000.0 << " ms, min: "
                << min_time_us / 1000.0 << " ms, max: "
                << max_time_us / 1000.0 << " ms\n";
    }}

    if (tpuRtStreamSynchronize(stream) != tpuRtSuccess) {{
      status = -8;
      break;
    }}
{memcpy_d2s_statements}
    if (status != 0) {{
      break;
    }}
  }} while (false);

{free_statements}
  post();
  return status;
}}
