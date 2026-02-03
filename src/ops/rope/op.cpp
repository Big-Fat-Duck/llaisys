#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");

    // in/out: [seqlen, nhead, d]  (or nkvhead)
    ASSERT(in->ndim() == 3 && out->ndim() == 3, "RoPE: in/out must be 3D.");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D.");

    const size_t seqlen = in->shape()[0];
    const size_t nhead  = in->shape()[1];
    const size_t d      = in->shape()[2];

    ASSERT(out->shape()[0] == seqlen && out->shape()[1] == nhead && out->shape()[2] == d,
           "RoPE: out shape mismatch.");
    ASSERT(pos_ids->shape()[0] == seqlen, "RoPE: pos_ids length must equal seqlen.");
    ASSERT(d % 2 == 0, "RoPE: last dim d must be even.");

    ASSERT(in->isContiguous() && out->isContiguous() && pos_ids->isContiguous(),
           "RoPE: tensors must be contiguous for now.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            out->dtype(),
            seqlen,
            nhead,
            d,
            theta
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         out->dtype(), seqlen, nhead, d, theta);
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
