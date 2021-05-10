#pragma once
#include <kungfu/nccl/helper.hpp>

extern std::unique_ptr<kungfu::NCCLHelper> _kungfu_nccl_helper;

#define KF_LOG_CALL(e)                                                         \
    log_func_call(#e);                                                         \
    e

extern void kungfu_show_cuda_version();
extern void kungfu_show_nccl_version();

template <typename T>
kungfu::Workspace make_kungfu_workspace(const T *input, T *output, int count)
{
    return {
        .sendbuf = input,
        .recvbuf = output,
        .count = count,
        .dtype = kungfu::type_encoder::value<T>(),
    };
}

namespace mindspore
{
namespace kernel
{
extern bool kungfu_use_nccl_scheduler;

extern void init_kungfu_nccl_once();
extern void finalize_kungfu_nccl();
}  // namespace kernel
}  // namespace   mindspore
