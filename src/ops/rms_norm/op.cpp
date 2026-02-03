#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    ASSERT(in->ndim() == 2 && out->ndim() == 2, "RMSNorm: in/out must be 2D.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D.");

    const size_t M = in->shape()[0];
    const size_t D = in->shape()[1];

    ASSERT(out->shape()[0] == M && out->shape()[1] == D, "RMSNorm: out shape mismatch.");
    ASSERT(weight->shape()[0] == D, "RMSNorm: weight length must equal last dim of input.");

    ASSERT(in->isContiguous() && out->isContiguous() && weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous for now.");

    // CPU fast path
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            M,
            D,
            eps
        );
    }

    // device dispatch
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), M, D, eps);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
