#include "backend/kernel_compiler/gpu/kungfu/kungfu_all_reduce_gpu_kernel.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"

namespace mindspore
{
namespace kernel
{
MS_REG_GPU_KERNEL_ONE(KungFuAllReduce,
                      KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat32),
                      KungFuAllReduceGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
    KungFuAllReduce,
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KungFuAllReduceGpuKernel, int32_t)
}  // namespace kernel
}  // namespace   mindspore
