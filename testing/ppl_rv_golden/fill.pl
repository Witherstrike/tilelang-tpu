#include "ppl.h"
using namespace ppl;

__KERNEL__ void fill_kernel(fp16 *out) {
  dim4 shape = {1, 8, 1, 16};
  auto g_out = gtensor<fp16>(shape, GLOBAL, out);
  tensor<fp16> local(shape);
  tiu::fill(local, 1.5f);
  dma::store(g_out, local);
}
