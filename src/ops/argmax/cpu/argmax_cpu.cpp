#include "argmax_cpu.hpp"

#include "../../../utils.hpp"
#include <cstdint>
#include <type_traits>

template <typename T>
static void argmax_(int64_t* out_idx, T* out_val, const T* vals, size_t numel) {
    // TODO: implement
    // - handle numel > 0
    // - scan vals[0..numel-1]
    // - write best index to *out_idx
    // - write best value to *out_val
    ASSERT(numel > 0, "Argmax: vals is empty.");
    float mx = llaisys::utils::cast<float>(vals[0]);
    int64_t p = 0;
    for (size_t i = 1; i < numel; ++i) {
        if (llaisys::utils::cast<float>(vals[i]) > mx) {
            p = i;
            mx = llaisys::utils::cast<float>(vals[i]);
        }
    }
    *out_val = vals[p];
    *out_idx = p;
}

namespace llaisys::ops::cpu {

void argmax(std::byte* max_idx,
            std::byte* max_val,
            const std::byte* vals,
            llaisysDataType_t type,
            size_t numel) {

    auto* out_idx = reinterpret_cast<int64_t*>(max_idx);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(out_idx,
                       reinterpret_cast<float*>(max_val),
                       reinterpret_cast<const float*>(vals),
                       numel);

    case LLAISYS_DTYPE_F16:
        return argmax_(out_idx,
                       reinterpret_cast<llaisys::fp16_t*>(max_val),
                       reinterpret_cast<const llaisys::fp16_t*>(vals),
                       numel);

    case LLAISYS_DTYPE_BF16:
        return argmax_(out_idx,
                       reinterpret_cast<llaisys::bf16_t*>(max_val),
                       reinterpret_cast<const llaisys::bf16_t*>(vals),
                       numel);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
