#include <iostream>

#include <kungfu.h>
#include <kungfu/logger.hpp>
#include <kungfu/nccl/helper.hpp>

#include <nccl.h>

#include "backend/kernel_compiler/cpu/kungfu/kungfu_common.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"
#include "pybind_api/api_register.h"

class Tracer
{
    std::string name_;

    static int Indent(int x = 0)
    {
        static int indent = 0;
        indent += x;
        return indent;
    }

    static std::string tab()
    {
        return std::string(Indent() * 4, ' ');
    }

  public:
    Tracer(std::string name) : name_(std::move(name))
    {
        std::cout << tab() << "{ //begin " << name_ << std::endl;
        Indent(1);
    }

    ~Tracer()
    {
        Indent(-1);
        std::cout << tab() << "} // end " << name_ << std::endl;
    }
};

#define KF_TRACE(name) Tracer __trace(name);

template <typename T>
class cuda_vector
{
    T *ptr_;

  public:
    cuda_vector(size_t n)
    {
        auto ret = cudaMalloc(&ptr_, n * sizeof(T));
        if (ret != cudaSuccess) {
            MS_LOG(ERROR) << "cudaMalloc failed, ret[" << static_cast<int>(ret)
                          << "], " << cudaGetErrorString(ret);
        }
    }

    ~cuda_vector()
    {
        auto ret = cudaFree(ptr_);
        if (ret != cudaSuccess) {
            MS_LOG(ERROR) << "cudaFree failed, ret[" << static_cast<int>(ret)
                          << "], " << cudaGetErrorString(ret);
        }
    }

    T *data()
    {
        return ptr_;
    }
};

void kungfu_debug_nccl()
{
    KF_TRACE(__func__);
    kungfu_show_cuda_version();
    kungfu_show_nccl_version();

    auto rank = _kungfu_peer->Rank();
    auto size = _kungfu_peer->Size();
    KF_LOG() << "rank:" << rank;
    KF_LOG() << "size:" << size;

    size_t n = 1024;
    using T = float;
    cuda_vector<T> x(n);
    cuda_vector<T> y(n);
    auto w = make_kungfu_workspace(x.data(), y.data(), n);

    kungfu::NCCLController nccl(KungFu_NCCL_GLOBAL);
    nccl.InitOnce(_kungfu_peer.get());
    {
        KF_TRACE("nccl.AllReduce");
        nccl.AllReduce(w, KungFu_SUM, [] {});
    }
}

namespace mindspore
{
namespace kernel
{
REGISTER_PYBIND_DEFINE(KungfuDebug, ([](py::module *m) {
                           m->def("kungfu_debug_nccl", &kungfu_debug_nccl);
                       }));
}
}  // namespace   mindspore
