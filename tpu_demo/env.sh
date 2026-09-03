export SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/tl_templates/tpu"
if [ -z "${PPL_PROJECT_ROOT:-}" ]; then
  echo "PPL_PROJECT_ROOT must point to a PPL 1.7 SDK" >&2
  return 1 2>/dev/null || exit 1
fi
export LD_LIBRARY_PATH="${PPL_PROJECT_ROOT}/deps/runtime/tpuv7-runtime/lib:${LD_LIBRARY_PATH:-}"
export TPU_KERNEL_PATH="${SRC_DIR}"
export PPL_KERNEL_PATH="${SRC_DIR}/libkernel.so"
