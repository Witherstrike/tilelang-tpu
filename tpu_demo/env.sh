if [ -z "${PPL_PROJECT_ROOT:-}" ]; then
  echo "PPL_PROJECT_ROOT must point to a PPL 1.7 SDK" >&2
  return 1 2>/dev/null || exit 1
fi
export LD_LIBRARY_PATH="${PPL_PROJECT_ROOT}/deps/runtime/tpuv7-runtime/lib:${LD_LIBRARY_PATH:-}"

# TileLang JIT embeds the matching private libkernel.so path while compiling;
# do not export a shared PPL_KERNEL_PATH or TPU_KERNEL_PATH here. A manual
# standalone host binary, if one explicitly uses that legacy environment
# contract, must set its own absolute PPL_KERNEL_PATH outside this script.
