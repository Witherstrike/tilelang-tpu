import tilelang
import tilelang.language as T
import torch


def topk_kernel(length, K, descended=True, dtype="float32"):

    @T.prim_func
    def main_kernel_inner(
            Input: T.Tensor((length,), dtype),
            Output: T.Tensor((K,), dtype),
            Indices: T.Tensor((K,), "int32"),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (bx, by):
            T.ppl_topk(Output, Indices, Input, K, descended, length)

    return main_kernel_inner


LENGTH = 1024
K = 8

kernel = tilelang.compile(
    topk_kernel(LENGTH, K, descended=True),
    out_idx=[1, 2],
    target="tpu -mcpu=bm1690 -tpu-programming-model=tpukernel",
)

src = torch.randn(LENGTH).float()
output = torch.zeros(K).float()
indices = torch.zeros(K, dtype=torch.int32)
res = kernel(src, output, indices)

topk_vals = output
topk_idx = indices

ref_vals, ref_idx = torch.topk(src, K, largest=True, sorted=True)

val_diff = (topk_vals - ref_vals).abs()
max_diff = val_diff.max().item()
avg_diff = val_diff.mean().item()
print(f"top-{K} values  (TPU): {topk_vals.tolist()}")
print(f"top-{K} values  (ref): {ref_vals.tolist()}")
print(f"top-{K} indices (TPU): {topk_idx.tolist()}")
print(f"top-{K} indices (ref): {ref_idx.tolist()}")
print("\n=== 差异分析 ===")
print(f"value 最大差异: {max_diff:.6f}")
print(f"value 平均差异: {avg_diff:.6f}")

tpu_vals_via_idx = src[topk_idx.long()]
print(f"index 反查值 allclose: {torch.allclose(tpu_vals_via_idx, ref_vals, atol=1e-5)}")
print(f"value allclose (atol=1e-5): {torch.allclose(topk_vals, ref_vals, atol=1e-5)}")
