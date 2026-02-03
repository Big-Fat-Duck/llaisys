#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>   // std::memcpy
#include <type_traits>

template <typename T>
static void embedding_(T* out,
                       const int64_t* idx,
                       const T* weight,
                       size_t n,
                       size_t vocab,
                       size_t dim) {
    // vocab 有几个向量
    // dim 每个向量有几个元素（几维）？
    // n 一共有几个 idx
    // weight 输入的二维数组
    // out 输出的二维数组

    for (size_t i = 0; i < n; i++) ASSERT(0 <= idx[i] && (size_t)idx[i] < vocab, "Embedding: index out of range");
    for (size_t i = 0; i < n; ++i) {
        int64_t row = idx[i];
        for (size_t j = 0; j < dim; j++) out[i * dim + j] = weight[(size_t)row * dim + j];
    }
}

namespace llaisys::ops::cpu {

void embedding(std::byte* out,
               const std::byte* index,
               const std::byte* weight,
               llaisysDataType_t type,
               size_t n,
               size_t vocab,
               size_t dim) {
    const auto* idx = reinterpret_cast<const int64_t*>(index);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(
            reinterpret_cast<float*>(out),
            idx,
            reinterpret_cast<const float*>(weight),
            n, vocab, dim
        );
    case LLAISYS_DTYPE_F16:
        return embedding_(
            reinterpret_cast<llaisys::fp16_t*>(out),
            idx,
            reinterpret_cast<const llaisys::fp16_t*>(weight),
            n, vocab, dim
        );
    case LLAISYS_DTYPE_BF16:
        return embedding_(
            reinterpret_cast<llaisys::bf16_t*>(out),
            idx,
            reinterpret_cast<const llaisys::bf16_t*>(weight),
            n, vocab, dim
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
