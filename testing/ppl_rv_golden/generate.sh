#!/usr/bin/env bash
set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ppl_root=${PPL_PROJECT_ROOT:?Set PPL_PROJECT_ROOT to the PPL 1.7 release root}
output_root=${1:-"$script_dir/artifacts"}
ops=(copy fill elementwise gemm reduction special_function gather topk)
failed=0

for op in "${ops[@]}"; do
  output_dir="$output_root/$op"
  mkdir -p "$output_dir"
  echo "[PPL RV] $op"
  "$ppl_root/bin/ppl-compile" "$script_dir/$op.pl" \
    -I "$ppl_root/inc" \
    --chip sg2260e \
    -D__sg2260e__ \
    --rv --O3 --print-ir \
    -o "$output_dir" >"$output_dir/compile.log" 2>&1
  status=$?
  if [[ $op == topk && $status -eq 23 ]]; then
    echo "  expected PPL 1.7 limitation: rc=23"
  elif [[ $status -ne 0 ]]; then
    echo "  unexpected failure: rc=$status"
    failed=1
  else
    echo "  success"
  fi
done

exit "$failed"
