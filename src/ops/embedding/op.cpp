#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);

    // shape checks (作业假设：index 1D, weight 2D, out 2D)
    ASSERT(index->ndim() == 1, "Embedding: index must be 1D.");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    ASSERT(out->ndim() == 2, "Embedding: out must be 2D.");

    // dtype checks
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index dtype must be int64.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    // shape relation: out = weight[index]
    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: out.shape[0] must equal index.shape[0].");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: out.shape[1] must equal weight.shape[1].");

    // contiguous for now
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    // CPU fast path
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            out->dtype(),
            /*n=*/index->shape()[0],
            /*vocab=*/weight->shape()[0],
            /*dim=*/weight->shape()[1]
        );
    }

    // device dispatch
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            out->data(), index->data(), weight->data(), out->dtype(),
            index->shape()[0], weight->shape()[0], weight->shape()[1]
        );

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
