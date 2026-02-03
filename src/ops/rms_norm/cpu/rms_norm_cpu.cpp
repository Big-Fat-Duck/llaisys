#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <type_traits>

template <typename T>
static void rms_norm_(T* out,
                      const T* in,
                      const T* w,     // weight [D]
                      size_t M,
                      size_t D,
                      float eps) {
    // TODO: implement
    // for each row i:
    //   mean_sq = (1/D) * sum_j (x^2)
    //   inv_rms = 1 / sqrt(mean_sq + eps)
    //   out[i,j] = w[j] * x[i,j] * inv_rms

    for (size_t i = 0; i < M; ++i) {
        // 1) sum of squares
        float sqr_sum = 0;
        for (size_t j = 0; j < D; j++) 
            sqr_sum += llaisys::utils::cast<float>(in[i * D + j]) * 
                       llaisys::utils::cast<float>(in[i * D + j]);
        
        // 2) inv rms
        float fen_mu = sqrt(sqr_sum / D + eps);

        // 3) normalize and scale
        for (size_t j = 0; j < D; j++) 
            out[i * D + j] = llaisys::utils::cast<T>(
                                llaisys::utils::cast<float>(in[i * D + j]) * 
                                llaisys::utils::cast<float>(w[j]) 
                                / fen_mu);
    }
}

namespace llaisys::ops::cpu {

void rms_norm(std::byte* out,
              const std::byte* in,
              const std::byte* weight,
              llaisysDataType_t type,
              size_t M,
              size_t D,
              float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(in),
            reinterpret_cast<const float*>(weight),
            M, D, eps
        );
    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t*>(out),
            reinterpret_cast<const llaisys::fp16_t*>(in),
            reinterpret_cast<const llaisys::fp16_t*>(weight),
            M, D, eps
        );
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t*>(out),
            reinterpret_cast<const llaisys::bf16_t*>(in),
            reinterpret_cast<const llaisys::bf16_t*>(weight),
            M, D, eps
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
