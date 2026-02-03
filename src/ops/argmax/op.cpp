#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // vals 需要找最大值的那个数组
    // max_idx 最大值位置的索引
    // max_val 最大值的“值”
    // ---- checks ----
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // 作业要求：暂时只支持 1D vals，max_idx/max_val 是 1D 且只有 1 个元素
    ASSERT(vals->ndim() == 1, "Argmax: vals must be a 1D tensor for now.");
    ASSERT(max_idx->ndim() == 1 && max_idx->numel() == 1, "Argmax: max_idx must be 1D with 1 element.");
    ASSERT(max_val->ndim() == 1 && max_val->numel() == 1, "Argmax: max_val must be 1D with 1 element.");

    // contiguous（先按最简单版本）
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(),
           "Argmax: all tensors must be contiguous.");

    // dtype 约束：index 输出通常是 int64；max_val 和 vals 同 dtype
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx dtype must be int64.");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    // ---- CPU fast path ----
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel()
        );
    }

    // ---- device dispatch ----
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel()
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
