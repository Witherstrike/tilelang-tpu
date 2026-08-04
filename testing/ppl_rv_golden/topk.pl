#include "ppl.h"
using namespace ppl;

__KERNEL__ void topk_kernel(fp32 *out, int *out_index, fp32 *in) {
  dim4 in_shape = {1, 1, 1, 16};
  dim4 out_shape = {1, 1, 1, 4};
  auto g_in = gtensor<fp32>(in_shape, GLOBAL, in);
  auto g_out = gtensor<fp32>(out_shape, GLOBAL, out);
  auto g_out_index = gtensor<int>(out_shape, GLOBAL, out_index);
  hau::topk(g_out, g_out_index, g_in, 4, true);
}
