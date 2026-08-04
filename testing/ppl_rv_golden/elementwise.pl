#include "ppl.h"
using namespace ppl;

__KERNEL__ void elementwise_kernel(fp16 *out, fp16 *lhs, fp16 *rhs) {
  dim4 shape = {1, 8, 1, 16};
  auto g_lhs = gtensor<fp16>(shape, GLOBAL, lhs);
  auto g_rhs = gtensor<fp16>(shape, GLOBAL, rhs);
  auto g_out = gtensor<fp16>(shape, GLOBAL, out);
  tensor<fp16> l(shape), r(shape), sum(shape), result(shape);
  dma::load(l, g_lhs);
  dma::load(r, g_rhs);
  tiu::fadd(sum, l, r);
  tiu::fmul(result, sum, 0.5f);
  dma::store(g_out, result);
}
