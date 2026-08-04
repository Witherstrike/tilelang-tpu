#include "ppl.h"
using namespace ppl;

__KERNEL__ void copy_kernel(fp16 *out, fp16 *in) {
  dim4 shape = {1, 8, 1, 16};
  auto g_in = gtensor<fp16>(shape, GLOBAL, in);
  auto g_out = gtensor<fp16>(shape, GLOBAL, out);
  tensor<fp16> local(shape);
  dma::load(local, g_in);
  dma::store(g_out, local);
}
