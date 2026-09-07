import tilelang
import tilelang.language as T


def kernel(length, K, descended=True, dtype="float32"):

    @T.prim_func
    def main_kernel_inner(
            Input: T.Tensor((length,), dtype),
            Output: T.Tensor((K,), dtype),
            Indices: T.Tensor((K,), "int32"),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (bx, by):
            T.ppl_topk(Output, Indices, Input, K, descended, length)

    return main_kernel_inner


func = kernel(length=16, K=4, descended=True, dtype="float32")
artifact = tilelang.lower(
    func,
    target="tpu -mcpu=bm1690 -tpu-programming-model=tpukernel",
    runtime_mode="cmodel",
)
print("\n\n\n")
print(artifact.kernel_source)
