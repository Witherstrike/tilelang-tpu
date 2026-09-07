#!/usr/bin/env bash
# Build the TileLang TPU backend without modifying the vendored TVM checkout.
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
build_dir="$script_dir/build"
config_file="$build_dir/config.cmake"

mkdir -p "$build_dir"
if [[ ! -f "$config_file" ]]; then
  cp "$script_dir/3rdparty/tvm/cmake/config.cmake" "$config_file"
fi

echo "Running CMake for TileLang TPU..."
cmake -S "$script_dir" -B "$build_dir"

# Avoid monopolising the host while still making a normal build reasonably fast.
cores="$(nproc)"
jobs=$((cores * 3 / 4))
if (( jobs < 1 )); then
  jobs=1
fi

echo "Building TileLang TPU with $jobs job(s)..."
cmake --build "$build_dir" --parallel "$jobs"
echo "TileLang TPU build completed successfully."
