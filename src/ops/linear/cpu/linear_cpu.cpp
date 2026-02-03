#include "linear_cpu.hpp"

#include "../../../utils.hpp"
#include <cstdint>
#include <type_traits>

template <typename T>
static void linear_(T* out,
                    const T* in,
                    const T* w,           // weight [N,K]
                    const T* bias,        // optional [N], can be nullptr
                    size_t M, size_t N, size_t K) {
    // TODO: implement (naive triple loop)
    // out[i*N + j] = sum_k in[i*K + k] * w[j*K + k] + (bias? bias[j] : 0)

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            // TODO: if bias != nullptr: acc = cast<float>(bias[j])
            if (bias) acc = llaisys::utils::cast<float>(bias[j]);
            for (size_t k = 0; k < K; ++k) {
                acc += llaisys::utils::cast<float>(in[i * K + k]) * llaisys::utils::cast<float>(w[j * K + k]);
            }
            out[i * N + j] = llaisys::utils::cast<T>(acc);
        }
    }
}

namespace llaisys::ops::cpu {

void linear(std::byte* out,
            const std::byte* in,
            const std::byte* weight,
            const std::byte* bias,   // nullable
            llaisysDataType_t type,
            size_t M, size_t N, size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(in),
            reinterpret_cast<const float*>(weight),
            bias ? reinterpret_cast<const float*>(bias) : nullptr,
            M, N, K
        );
    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t*>(out),
            reinterpret_cast<const llaisys::fp16_t*>(in),
            reinterpret_cast<const llaisys::fp16_t*>(weight),
            bias ? reinterpret_cast<const llaisys::fp16_t*>(bias) : nullptr,
            M, N, K
        );
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t*>(out),
            reinterpret_cast<const llaisys::bf16_t*>(in),
            reinterpret_cast<const llaisys::bf16_t*>(weight),
            bias ? reinterpret_cast<const llaisys::bf16_t*>(bias) : nullptr,
            M, N, K
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
