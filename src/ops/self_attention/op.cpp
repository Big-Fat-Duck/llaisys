#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace {

template <typename T>
void self_attention_cpu_(
    T *out,
    const T *q,
    const T *k,
    const T *v,
    size_t qlen,
    size_t kvlen,
    size_t nh,
    size_t nkvh,
    size_t d,
    size_t dv,
    float scale) {
    const size_t head_repeat = nh / nkvh;

    for (size_t qi = 0; qi < qlen; ++qi) {
        const size_t max_k = (kvlen - qlen) + qi;

        for (size_t h = 0; h < nh; ++h) {
            const size_t kvh = h / head_repeat;

            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t kj = 0; kj <= max_k; ++kj) {
                float dot = 0.0f;
                const size_t q_base = (qi * nh + h) * d;
                const size_t k_base = (kj * nkvh + kvh) * d;
                for (size_t t = 0; t < d; ++t) {
                    const float qv = llaisys::utils::cast<float>(q[q_base + t]);
                    const float kv = llaisys::utils::cast<float>(k[k_base + t]);
                    dot += qv * kv;
                }
                dot *= scale;
                if (dot > max_logit) max_logit = dot;
            }

            float sum = 0.0f;
            std::vector<float> out_acc(dv, 0.0f);
            for (size_t kj = 0; kj <= max_k; ++kj) {
                float dot = 0.0f;
                const size_t q_base = (qi * nh + h) * d;
                const size_t k_base = (kj * nkvh + kvh) * d;
                for (size_t t = 0; t < d; ++t) {
                    const float qv = llaisys::utils::cast<float>(q[q_base + t]);
                    const float kv = llaisys::utils::cast<float>(k[k_base + t]);
                    dot += qv * kv;
                }
                dot = dot * scale;
                const float w = std::exp(dot - max_logit);
                sum += w;

                const size_t v_base = (kj * nkvh + kvh) * dv;
                for (size_t t = 0; t < dv; ++t) {
                    const float vv = llaisys::utils::cast<float>(v[v_base + t]);
                    out_acc[t] += w * vv;
                }
            }

            const size_t out_base = (qi * nh + h) * dv;
            const float inv_sum = 1.0f / sum;
            for (size_t t = 0; t < dv; ++t) {
                out[out_base + t] = llaisys::utils::cast<T>(out_acc[t] * inv_sum);
            }
        }
    }
}

void self_attention_cpu(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t dtype,
    size_t qlen,
    size_t kvlen,
    size_t nh,
    size_t nkvh,
    size_t d,
    size_t dv,
    float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_cpu_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            qlen, kvlen, nh, nkvh, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_cpu_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            qlen, kvlen, nh, nkvh, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_cpu_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            qlen, kvlen, nh, nkvh, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "SelfAttention: all tensors must be 3D.");

    const size_t qlen  = q->shape()[0];
    const size_t nh    = q->shape()[1];
    const size_t d     = q->shape()[2];
    const size_t kvlen = k->shape()[0];
    const size_t nkvh  = k->shape()[1];
    const size_t kd    = k->shape()[2];
    const size_t vlen  = v->shape()[0];
    const size_t vkvh  = v->shape()[1];
    const size_t dv    = v->shape()[2];

    ASSERT(kd == d, "SelfAttention: q/k head dim mismatch.");
    ASSERT(vlen == kvlen && vkvh == nkvh, "SelfAttention: k/v shape mismatch.");
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == dv,
           "SelfAttention: attn_val shape mismatch.");
    ASSERT(nh % nkvh == 0, "SelfAttention: nh must be divisible by nkvh.");
    ASSERT(kvlen >= qlen, "SelfAttention: kvlen must be >= qlen.");

    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: tensors must be contiguous for now.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return self_attention_cpu(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), qlen, kvlen, nh, nkvh, d, dv, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return self_attention_cpu(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), qlen, kvlen, nh, nkvh, d, dv, scale);
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
