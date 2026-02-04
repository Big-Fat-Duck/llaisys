#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;

    // 你后面会加：KV cache、临时buffer、device信息等
};

static void* xcalloc(size_t n, size_t sz) {
    void* p = std::calloc(n, sz);
    return p;
}

static llaisysTensor_t make_1d(
    llaisysDataType_t dt,
    llaisysDeviceType_t dev,
    int dev_id,
    size_t a
) {
    size_t shape[1] = {a};
    return tensorCreate(shape, 1, dt, dev, dev_id);
}

static llaisysTensor_t make_2d(
    llaisysDataType_t dt,
    llaisysDeviceType_t dev,
    int dev_id,
    size_t a,
    size_t b
) {
    size_t shape[2] = {a, b};
    return tensorCreate(shape, 2, dt, dev, dev_id);
}

static llaisysTensor_t make_3d(
    llaisysDataType_t dt,
    llaisysDeviceType_t dev,
    int dev_id,
    size_t a,
    size_t b,
    size_t c
) {
    size_t shape[3] = {a, b, c};
    return tensorCreate(shape, 3, dt, dev, dev_id);
}

extern "C" __export LlaisysQwen2Model* llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta* meta,
    llaisysDeviceType_t device,
    int* device_ids,
    int ndevice
) {
    int dev_id = (device_ids && ndevice > 0) ? device_ids[0] : 0;
    auto* m = (LlaisysQwen2Model*)std::calloc(1, sizeof(LlaisysQwen2Model));
    m->meta = *meta;

    const size_t L = meta->nlayer;
    const size_t hs = meta->hs;
    const size_t nh = meta->nh;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t di = meta->di;
    const size_t voc = meta->voc;

    const size_t q_out  = nh   * dh;   // Q 输出维
    const size_t kv_out = nkvh * dh;   // K/V 输出维

    // 1) 分配每层数组
    m->weights.attn_norm_w = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    m->weights.attn_q_w    = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    m->weights.attn_q_b    = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    m->weights.attn_k_w    = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    m->weights.attn_k_b    = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    m->weights.attn_v_w    = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    m->weights.attn_v_b    = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    m->weights.attn_o_w    = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));

    m->weights.mlp_norm_w  = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    m->weights.mlp_gate_w  = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    m->weights.mlp_up_w    = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    m->weights.mlp_down_w  = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));

    // 2) 创建全局权重句柄（之前为 0 的根因就在这）
    m->weights.in_embed   = make_2d(meta->dtype, device, dev_id, voc, hs);
    m->weights.out_embed  = make_2d(meta->dtype, device, dev_id, voc, hs);
    m->weights.out_norm_w = make_1d(meta->dtype, device, dev_id, hs);

    // 3) 创建每层权重句柄
    for (size_t i = 0; i < L; ++i) {
        m->weights.attn_norm_w[i] = make_1d(meta->dtype, device, dev_id, hs);

        m->weights.attn_q_w[i] = make_2d(meta->dtype, device, dev_id, q_out,  hs);
        m->weights.attn_k_w[i] = make_2d(meta->dtype, device, dev_id, kv_out, hs);
        m->weights.attn_v_w[i] = make_2d(meta->dtype, device, dev_id, kv_out, hs);
        m->weights.attn_o_w[i] = make_2d(meta->dtype, device, dev_id, hs, q_out);

        m->weights.attn_q_b[i] = make_1d(meta->dtype, device, dev_id, q_out);
        m->weights.attn_k_b[i] = make_1d(meta->dtype, device, dev_id, kv_out);
        m->weights.attn_v_b[i] = make_1d(meta->dtype, device, dev_id, kv_out);

        m->weights.mlp_norm_w[i] = make_1d(meta->dtype, device, dev_id, hs);
        m->weights.mlp_gate_w[i] = make_2d(meta->dtype, device, dev_id, di, hs);
        m->weights.mlp_up_w[i]   = make_2d(meta->dtype, device, dev_id, di, hs);
        m->weights.mlp_down_w[i] = make_2d(meta->dtype, device, dev_id, hs, di);
    }

    return m;
}

extern "C" __export void llaisysQwen2ModelDestroy(LlaisysQwen2Model* m) {
    if (!m) return;
    std::free(m->weights.attn_norm_w);
    std::free(m->weights.attn_q_w);
    std::free(m->weights.attn_q_b);
    std::free(m->weights.attn_k_w);
    std::free(m->weights.attn_k_b);
    std::free(m->weights.attn_v_w);
    std::free(m->weights.attn_v_b);
    std::free(m->weights.attn_o_w);

    std::free(m->weights.mlp_norm_w);
    std::free(m->weights.mlp_gate_w);
    std::free(m->weights.mlp_up_w);
    std::free(m->weights.mlp_down_w);

    std::free(m);
}

extern "C" __export LlaisysQwen2Weights* llaisysQwen2ModelWeights(LlaisysQwen2Model* m) {
    return m ? &m->weights : nullptr;
}

extern "C" __export int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model* m, int64_t* token_ids, size_t ntoken) {
    if (!m || ntoken == 0) return m ? m->meta.end_token : 0;

    const LlaisysQwen2Meta& meta = m->meta;
    const size_t L = meta.nlayer;
    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;
    const size_t q_out = nh * dh;
    const size_t kv_out = nkvh * dh;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    const llaisysDeviceType_t dev = tensorGetDeviceType(m->weights.in_embed);
    const int dev_id = tensorGetDeviceId(m->weights.in_embed);
    const llaisysDataType_t dt = meta.dtype;

    llaisysTensor_t idx = make_1d(LLAISYS_DTYPE_I64, dev, dev_id, ntoken);
    tensorLoad(idx, token_ids);

    std::vector<int64_t> pos_host(ntoken);
    for (size_t i = 0; i < ntoken; ++i) pos_host[i] = static_cast<int64_t>(i);
    llaisysTensor_t pos_ids = make_1d(LLAISYS_DTYPE_I64, dev, dev_id, ntoken);
    tensorLoad(pos_ids, pos_host.data());

    llaisysTensor_t x = make_2d(dt, dev, dev_id, ntoken, hs);
    llaisysEmbedding(x, idx, m->weights.in_embed);

    llaisysTensor_t norm1 = make_2d(dt, dev, dev_id, ntoken, hs);
    llaisysTensor_t norm2 = make_2d(dt, dev, dev_id, ntoken, hs);
    llaisysTensor_t q_lin = make_2d(dt, dev, dev_id, ntoken, q_out);
    llaisysTensor_t k_lin = make_2d(dt, dev, dev_id, ntoken, kv_out);
    llaisysTensor_t v_lin = make_2d(dt, dev, dev_id, ntoken, kv_out);
    llaisysTensor_t attn = make_3d(dt, dev, dev_id, ntoken, nh, dh);
    llaisysTensor_t attn_proj = make_2d(dt, dev, dev_id, ntoken, hs);
    llaisysTensor_t gate = make_2d(dt, dev, dev_id, ntoken, di);
    llaisysTensor_t up = make_2d(dt, dev, dev_id, ntoken, di);
    llaisysTensor_t swiglu = make_2d(dt, dev, dev_id, ntoken, di);
    llaisysTensor_t down = make_2d(dt, dev, dev_id, ntoken, hs);

    size_t q_shape[3] = {ntoken, nh, dh};
    size_t k_shape[3] = {ntoken, nkvh, dh};
    size_t attn2d_shape[2] = {ntoken, q_out};
    llaisysTensor_t q = tensorView(q_lin, q_shape, 3);
    llaisysTensor_t k = tensorView(k_lin, k_shape, 3);
    llaisysTensor_t v = tensorView(v_lin, k_shape, 3);
    llaisysTensor_t attn2d = tensorView(attn, attn2d_shape, 2);

    for (size_t i = 0; i < L; ++i) {
        llaisysRmsNorm(norm1, x, m->weights.attn_norm_w[i], meta.epsilon);

        llaisysLinear(q_lin, norm1, m->weights.attn_q_w[i], m->weights.attn_q_b[i]);
        llaisysLinear(k_lin, norm1, m->weights.attn_k_w[i], m->weights.attn_k_b[i]);
        llaisysLinear(v_lin, norm1, m->weights.attn_v_w[i], m->weights.attn_v_b[i]);

        llaisysROPE(q, q, pos_ids, meta.theta);
        llaisysROPE(k, k, pos_ids, meta.theta);

        llaisysSelfAttention(attn, q, k, v, scale);
        llaisysLinear(attn_proj, attn2d, m->weights.attn_o_w[i], nullptr);
        llaisysAdd(x, x, attn_proj);

        llaisysRmsNorm(norm2, x, m->weights.mlp_norm_w[i], meta.epsilon);
        llaisysLinear(gate, norm2, m->weights.mlp_gate_w[i], nullptr);
        llaisysLinear(up, norm2, m->weights.mlp_up_w[i], nullptr);
        llaisysSwiGLU(swiglu, gate, up);
        llaisysLinear(down, swiglu, m->weights.mlp_down_w[i], nullptr);
        llaisysAdd(x, x, down);
    }

    llaisysRmsNorm(x, x, m->weights.out_norm_w, meta.epsilon);

    llaisysTensor_t last = tensorSlice(x, 0, ntoken - 1, ntoken);
    llaisysTensor_t logits = make_2d(dt, dev, dev_id, 1, voc);
    llaisysLinear(logits, last, m->weights.out_embed, nullptr);

    size_t logits_shape[1] = {voc};
    llaisysTensor_t logits_1d = tensorView(logits, logits_shape, 1);
    llaisysTensor_t max_idx = make_1d(LLAISYS_DTYPE_I64, dev, dev_id, 1);
    llaisysTensor_t max_val = make_1d(dt, dev, dev_id, 1);
    llaisysArgmax(max_idx, max_val, logits_1d);

    int64_t next_token = *reinterpret_cast<int64_t*>(tensorGetData(max_idx));

    tensorDestroy(max_val);
    tensorDestroy(max_idx);
    tensorDestroy(logits_1d);
    tensorDestroy(logits);
    tensorDestroy(last);

    tensorDestroy(attn2d);
    tensorDestroy(v);
    tensorDestroy(k);
    tensorDestroy(q);

    tensorDestroy(down);
    tensorDestroy(swiglu);
    tensorDestroy(up);
    tensorDestroy(gate);
    tensorDestroy(attn_proj);
    tensorDestroy(attn);
    tensorDestroy(v_lin);
    tensorDestroy(k_lin);
    tensorDestroy(q_lin);
    tensorDestroy(norm2);
    tensorDestroy(norm1);
    tensorDestroy(x);
    tensorDestroy(pos_ids);
    tensorDestroy(idx);

    return next_token;
}
