#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace {

template <typename T>
void swiglu_cpu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        const float g = llaisys::utils::cast<float>(gate[i]);
        const float u = llaisys::utils::cast<float>(up[i]);
        const float sig = 1.0f / (1.0f + std::exp(-g));
        out[i] = llaisys::utils::cast<T>(u * g * sig);
    }
}

void swiglu_cpu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_cpu_(reinterpret_cast<float *>(out),
                           reinterpret_cast<const float *>(gate),
                           reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_cpu_(reinterpret_cast<llaisys::fp16_t *>(out),
                           reinterpret_cast<const llaisys::fp16_t *>(gate),
                           reinterpret_cast<const llaisys::fp16_t *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_cpu_(reinterpret_cast<llaisys::bf16_t *>(out),
                           reinterpret_cast<const llaisys::bf16_t *>(gate),
                           reinterpret_cast<const llaisys::bf16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());

    ASSERT(out->ndim() == 2 && gate->ndim() == 2 && up->ndim() == 2,
           "SwiGLU: out/gate/up must be 2D.");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: tensors must be contiguous for now.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return swiglu_cpu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return swiglu_cpu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
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
