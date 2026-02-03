#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias);

    // shape: in [M,K], weight [N,K], out [M,N]
    ASSERT(in->ndim() == 2 && weight->ndim() == 2 && out->ndim() == 2, "Linear: in/weight/out must be 2D.");

    const size_t M = in->shape()[0];
    const size_t K = in->shape()[1];
    const size_t N = weight->shape()[0];

    ASSERT(weight->shape()[1] == K, "Linear: weight shape mismatch (expect [N,K]).");
    ASSERT(out->shape()[0] == M && out->shape()[1] == N, "Linear: out shape mismatch (expect [M,N]).");

    // dtype: out/in/weight same; bias optional but must match out dtype
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == N, "Linear: bias must be 1D with shape [N].");
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    // contiguous for now
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: out/in/weight must be contiguous.");
    if (bias) ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");

    // CPU fast path
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias ? bias->data() : nullptr,
            out->dtype(),
            M, N, K
        );
    }

    // device dispatch
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias ? bias->data() : nullptr, out->dtype(), M, N, K);

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
