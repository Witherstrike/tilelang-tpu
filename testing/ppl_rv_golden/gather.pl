#include "ppl.h"
using namespace ppl;

__KERNEL__ void gather_kernel(fp16 *out, fp16 *table, uint32 *index) {
  dim4 table_shape = {1, 1, 16, 16};
  dim4 index_shape = {1, 1, 4, 1};
  dim4 out_shape = {1, 1, 4, 16};
  auto g_table = gtensor<fp16>(table_shape, GLOBAL, table);
  auto g_index = gtensor<uint32>(index_shape, GLOBAL, index);
  auto g_out = gtensor<fp16>(out_shape, GLOBAL, out);
  dma::gather_h(g_out, g_table, g_index, 0);
}
