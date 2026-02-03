#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cstring>
#include <vector>

namespace {

void rearrange_cpu(std::byte *out,
                   const std::byte *in,
                   const std::vector<size_t> &shape,
                   const std::vector<ptrdiff_t> &out_strides,
                   const std::vector<ptrdiff_t> &in_strides,
                   size_t elem_size) {
    const size_t ndim = shape.size();
    if (ndim == 0) return;

    size_t total = 1;
    for (size_t d = 0; d < ndim; ++d) total *= shape[d];

    std::vector<size_t> idx(ndim, 0);
    for (size_t it = 0; it < total; ++it) {
        ptrdiff_t out_off = 0;
        ptrdiff_t in_off = 0;
        for (size_t d = 0; d < ndim; ++d) {
            out_off += static_cast<ptrdiff_t>(idx[d]) * out_strides[d];
            in_off += static_cast<ptrdiff_t>(idx[d]) * in_strides[d];
        }
        std::memcpy(out + static_cast<size_t>(out_off) * elem_size,
                    in + static_cast<size_t>(in_off) * elem_size,
                    elem_size);

        for (size_t d = ndim; d-- > 0;) {
            idx[d]++;
            if (idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
}

} // namespace

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return rearrange_cpu(out->data(), in->data(), out->shape(),
                             out->strides(), in->strides(), out->elementSize());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return rearrange_cpu(out->data(), in->data(), out->shape(),
                             out->strides(), in->strides(), out->elementSize());
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
