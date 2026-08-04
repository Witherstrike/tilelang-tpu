#include "ppl.h"
using namespace ppl;

__KERNEL__ void reduction_kernel(fp16 *out, fp16 *in) {
  dim4 in_shape = {1, 8, 1, 16};
  dim4 out_shape = {1, 8, 1, 1};
  auto g_in = gtensor<fp16>(in_shape, GLOBAL, in);
  auto g_out = gtensor<fp16>(out_shape, GLOBAL, out);
  tensor<fp16> local_in(in_shape), local_out(out_shape);
  dma::load(local_in, g_in);
  tiu::reduce(local_out, local_in, ALL_REDUCE_SUM);
  dma::store(g_out, local_out);
}
