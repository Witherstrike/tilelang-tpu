#include "ppl.h"
using namespace ppl;

__KERNEL__ void gemm_kernel(fp16 *out, fp16 *lhs, fp16 *rhs) {
  dim4 lhs_shape = {1, 16, 1, 16};
  dim4 rhs_shape = {1, 16, 1, 16};
  dim4 out_shape = {1, 16, 1, 16};
  auto g_lhs = gtensor<fp16>(lhs_shape, GLOBAL, lhs);
  auto g_rhs = gtensor<fp16>(rhs_shape, GLOBAL, rhs);
  auto g_out = gtensor<fp16>(out_shape, GLOBAL, out);
  tensor<fp16> l(lhs_shape), r(rhs_shape), result(out_shape);
  dma::load(l, g_lhs);
  dma::load(r, g_rhs);
  tiu::fmm2_nn(result, l, r, false);
  dma::store(g_out, result);
}
