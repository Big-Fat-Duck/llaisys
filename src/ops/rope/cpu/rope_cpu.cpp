#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <type_traits>

template <typename T>
static void rope_(T* out,
                  const T* in,
                  const int64_t* pos,
                  size_t seqlen,
                  size_t nhead,
                  size_t d,
                  float theta) {
    // TODO: implement
    // in/out layout: [seqlen, nhead, d] contiguous
    // split last dim into two halves: a[0..d/2-1], b[0..d/2-1]
    // phi = p / theta^(2j/d)

    const size_t half = d / 2;

    for (size_t i = 0; i < seqlen; ++i) {
        const float p = static_cast<float>(pos[i]);
        for (size_t h = 0; h < nhead; ++h) {
            T*       out_ptr = out + (i * nhead + h) * d;
            const T* in_ptr  = in  + (i * nhead + h) * d;

            for (size_t j = 0; j < half; ++j) {
                float pre_x = llaisys::utils::cast<float>(*(in_ptr + j)), pre_y = llaisys::utils::cast<float>(*(in_ptr + half + j));
                float Phi = p / powf(theta, j * 2.0f / d);
                float cur_x = pre_x * cosf(Phi) - pre_y * sinf(Phi), cur_y = pre_y * cosf(Phi) + pre_x * sinf(Phi);
                *(out_ptr + j) = llaisys::utils::cast<T>(cur_x); *(out_ptr + half + j) = llaisys::utils::cast<T>(cur_y);
            }
        }
    }
}

namespace llaisys::ops::cpu {

void rope(std::byte* out,
          const std::byte* in,
          const std::byte* pos_ids,
          llaisysDataType_t type,
          size_t seqlen,
          size_t nhead,
          size_t d,
          float theta) {
    const auto* pos = reinterpret_cast<const int64_t*>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float*>(out),
                     reinterpret_cast<const float*>(in),
                     pos, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t*>(out),
                     reinterpret_cast<const llaisys::fp16_t*>(in),
                     pos, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t*>(out),
                     reinterpret_cast<const llaisys::bf16_t*>(in),
                     pos, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
