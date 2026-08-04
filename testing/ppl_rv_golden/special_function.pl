#include "ppl.h"
using namespace ppl;

__KERNEL__ void special_function_kernel(fp16 *out, fp16 *in) {
  dim4 shape = {1, 8, 1, 16};
  auto g_in = gtensor<fp16>(shape, GLOBAL, in);
  auto g_out = gtensor<fp16>(shape, GLOBAL, out);
  tensor<fp16> local_in(shape), local_out(shape);
  dma::load(local_in, g_in);
  tiu::frsqrt(local_out, local_in, 3);
  dma::store(g_out, local_out);
}
