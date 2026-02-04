#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"

static void* xcalloc(size_t n, size_t sz) {
    return std::calloc(n, sz);
}

static size_t dtype_nbytes(llaisysDataType_t dt) {
    switch (dt) {
    case LLAISYS_DTYPE_F16:  return 2;
    case LLAISYS_DTYPE_BF16: return 2;
    case LLAISYS_DTYPE_F32:  return 4;
    case LLAISYS_DTYPE_F64:  return 8;
    case LLAISYS_DTYPE_I64:  return 8;
    default: return 0;
    }
}

static llaisysTensor_t make_1d(llaisysDataType_t dt, llaisysDeviceType_t dev, int dev_id, size_t a) {
    size_t shape[1] = {a};
    return tensorCreate(shape, 1, dt, dev, dev_id);
}

static llaisysTensor_t make_2d(llaisysDataType_t dt, llaisysDeviceType_t dev, int dev_id, size_t a, size_t b) {
    size_t shape[2] = {a, b};
    return tensorCreate(shape, 2, dt, dev, dev_id);
}

static llaisysTensor_t make_3d(llaisysDataType_t dt, llaisysDeviceType_t dev, int dev_id, size_t a, size_t b, size_t c) {
    size_t shape[3] = {a, b, c};
    return tensorCreate(shape, 3, dt, dev, dev_id);
}

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;

    // runtime info
    llaisysDeviceType_t device;
    int dev_id;
    size_t seqlen; // 已经 cache 的 token 数（prefill 后等于 prompt_len）

    // KV cache: per-layer, shape [maxseq, nkvh, dh]
    llaisysTensor_t* k_cache;
    llaisysTensor_t* v_cache;

    // decode scratch (qlen = 1)
    llaisysTensor_t idx1;
    llaisysTensor_t pos1;
    llaisysTensor_t x1;
    llaisysTensor_t norm1;
    llaisysTensor_t norm2;
    llaisysTensor_t q_lin;
    llaisysTensor_t k_lin;
    llaisysTensor_t v_lin;
    llaisysTensor_t attn;
    llaisysTensor_t attn_proj;
    llaisysTensor_t gate;
    llaisysTensor_t up;
    llaisysTensor_t swiglu;
    llaisysTensor_t down;
    llaisysTensor_t logits;
    llaisysTensor_t max_idx;
    llaisysTensor_t max_val;

    // decode views
    llaisysTensor_t q;        // view of q_lin: [1, nh, dh]
    llaisysTensor_t k_step;   // view of k_lin: [1, nkvh, dh]
    llaisysTensor_t v_step;   // view of v_lin: [1, nkvh, dh]
    llaisysTensor_t attn2d;   // view of attn:  [1, q_out]
    llaisysTensor_t logits_1d;// view of logits:[voc]
};

extern "C" __export LlaisysQwen2Model* llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta* meta,
    llaisysDeviceType_t device,
    int* device_ids,
    int ndevice
) {
    int dev_id = (device_ids && ndevice > 0) ? device_ids[0] : 0;

    auto* m = (LlaisysQwen2Model*)std::calloc(1, sizeof(LlaisysQwen2Model));
    m->meta = *meta;
    m->device = device;
    m->dev_id = dev_id;
    m->seqlen = 0;

    const size_t L = meta->nlayer;
    const size_t hs = meta->hs;
    const size_t nh = meta->nh;
    const size_t nkvh = meta->nkvh;
    const size_t dh = meta->dh;
    const size_t di = meta->di;
    const size_t voc = meta->voc;

    const size_t q_out  = nh   * dh;   // Q 输出维
    const size_t kv_out = nkvh * dh;   // K/V 输出维

    // 1) 分配每层权重数组
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

    // 2) 创建全局权重句柄
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

    // 4) KV cache（CPU 先用 memcpy 方案，最少改动）
    m->k_cache = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    m->v_cache = (llaisysTensor_t*)xcalloc(L, sizeof(llaisysTensor_t));
    for (size_t i = 0; i < L; ++i) {
        m->k_cache[i] = make_3d(meta->dtype, device, dev_id, meta->maxseq, nkvh, dh);
        m->v_cache[i] = make_3d(meta->dtype, device, dev_id, meta->maxseq, nkvh, dh);
    }

    // 5) decode scratch（qlen=1）
    m->idx1 = make_1d(LLAISYS_DTYPE_I64, device, dev_id, 1);
    m->pos1 = make_1d(LLAISYS_DTYPE_I64, device, dev_id, 1);

    m->x1    = make_2d(meta->dtype, device, dev_id, 1, hs);
    m->norm1 = make_2d(meta->dtype, device, dev_id, 1, hs);
    m->norm2 = make_2d(meta->dtype, device, dev_id, 1, hs);

    m->q_lin = make_2d(meta->dtype, device, dev_id, 1, q_out);
    m->k_lin = make_2d(meta->dtype, device, dev_id, 1, kv_out);
    m->v_lin = make_2d(meta->dtype, device, dev_id, 1, kv_out);

    m->attn      = make_3d(meta->dtype, device, dev_id, 1, nh, dh);
    m->attn_proj = make_2d(meta->dtype, device, dev_id, 1, hs);

    m->gate   = make_2d(meta->dtype, device, dev_id, 1, di);
    m->up     = make_2d(meta->dtype, device, dev_id, 1, di);
    m->swiglu = make_2d(meta->dtype, device, dev_id, 1, di);
    m->down   = make_2d(meta->dtype, device, dev_id, 1, hs);

    m->logits  = make_2d(meta->dtype, device, dev_id, 1, voc);
    m->max_idx = make_1d(LLAISYS_DTYPE_I64, device, dev_id, 1);
    m->max_val = make_1d(meta->dtype, device, dev_id, 1);

    // views（只创建一次）
    {
        size_t q_shape[3] = {1, nh, dh};
        size_t kv_shape[3] = {1, nkvh, dh};
        size_t attn2d_shape[2] = {1, q_out};
        size_t logits_shape[1] = {voc};

        m->q = tensorView(m->q_lin, q_shape, 3);
        m->k_step = tensorView(m->k_lin, kv_shape, 3);
        m->v_step = tensorView(m->v_lin, kv_shape, 3);
        m->attn2d = tensorView(m->attn, attn2d_shape, 2);
        m->logits_1d = tensorView(m->logits, logits_shape, 1);
    }

    return m;
}

extern "C" __export void llaisysQwen2ModelDestroy(LlaisysQwen2Model* m) {
    if (!m) return;

    const size_t L = m->meta.nlayer;

    // destroy decode views
    tensorDestroy(m->logits_1d);
    tensorDestroy(m->attn2d);
    tensorDestroy(m->v_step);
    tensorDestroy(m->k_step);
    tensorDestroy(m->q);

    // destroy decode scratch
    tensorDestroy(m->max_val);
    tensorDestroy(m->max_idx);
    tensorDestroy(m->logits);
    tensorDestroy(m->down);
    tensorDestroy(m->swiglu);
    tensorDestroy(m->up);
    tensorDestroy(m->gate);
    tensorDestroy(m->attn_proj);
    tensorDestroy(m->attn);
    tensorDestroy(m->v_lin);
    tensorDestroy(m->k_lin);
    tensorDestroy(m->q_lin);
    tensorDestroy(m->norm2);
    tensorDestroy(m->norm1);
    tensorDestroy(m->x1);
    tensorDestroy(m->pos1);
    tensorDestroy(m->idx1);

    // destroy KV cache
    if (m->k_cache) {
        for (size_t i = 0; i < L; ++i) tensorDestroy(m->k_cache[i]);
        std::free(m->k_cache);
    }
    if (m->v_cache) {
        for (size_t i = 0; i < L; ++i) tensorDestroy(m->v_cache[i]);
        std::free(m->v_cache);
    }

    // destroy weight tensors
    tensorDestroy(m->weights.in_embed);
    tensorDestroy(m->weights.out_embed);
    tensorDestroy(m->weights.out_norm_w);
    for (size_t i = 0; i < L; ++i) {
        tensorDestroy(m->weights.attn_norm_w[i]);
        tensorDestroy(m->weights.attn_q_w[i]);
        tensorDestroy(m->weights.attn_q_b[i]);
        tensorDestroy(m->weights.attn_k_w[i]);
        tensorDestroy(m->weights.attn_k_b[i]);
        tensorDestroy(m->weights.attn_v_w[i]);
        tensorDestroy(m->weights.attn_v_b[i]);
        tensorDestroy(m->weights.attn_o_w[i]);

        tensorDestroy(m->weights.mlp_norm_w[i]);
        tensorDestroy(m->weights.mlp_gate_w[i]);
        tensorDestroy(m->weights.mlp_up_w[i]);
        tensorDestroy(m->weights.mlp_down_w[i]);
    }

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

static inline void kv_cache_store_cpu(
    llaisysTensor_t cache, size_t pos,
    const void* src,
    size_t row_elems,
    size_t elem_bytes
) {
    std::byte* dst = (std::byte*)tensorGetData(cache);
    std::memcpy(dst + pos * row_elems * elem_bytes, src, row_elems * elem_bytes);
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

    if (ntoken > meta.maxseq) return meta.end_token;

    const llaisysDeviceType_t dev = tensorGetDeviceType(m->weights.in_embed);
    const int dev_id = tensorGetDeviceId(m->weights.in_embed);
    const llaisysDataType_t dt = meta.dtype;

    const size_t elem_bytes = dtype_nbytes(dt);
    if (elem_bytes == 0) return meta.end_token;

    // ============================
    // 1) Prefill：第一次 or 非 +1 增长（比如换 prompt）
    // ============================
    if (m->seqlen == 0 || ntoken != m->seqlen + 1) {
        m->seqlen = 0;

        // idx [ntoken]
        llaisysTensor_t idx = make_1d(LLAISYS_DTYPE_I64, dev, dev_id, ntoken);
        tensorLoad(idx, token_ids);

        // pos_ids [ntoken]
        std::vector<int64_t> pos_host(ntoken);
        for (size_t i = 0; i < ntoken; ++i) pos_host[i] = static_cast<int64_t>(i);
        llaisysTensor_t pos_ids = make_1d(LLAISYS_DTYPE_I64, dev, dev_id, ntoken);
        tensorLoad(pos_ids, pos_host.data());

        // x [ntoken, hs]
        llaisysTensor_t x = make_2d(dt, dev, dev_id, ntoken, hs);
        llaisysEmbedding(x, idx, m->weights.in_embed);

        // tmp buffers (prefill)
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
        size_t kv_shape[3] = {ntoken, nkvh, dh};
        size_t attn2d_shape[2] = {ntoken, q_out};
        llaisysTensor_t q = tensorView(q_lin, q_shape, 3);
        llaisysTensor_t k = tensorView(k_lin, kv_shape, 3);
        llaisysTensor_t v = tensorView(v_lin, kv_shape, 3);
        llaisysTensor_t attn2d = tensorView(attn, attn2d_shape, 2);

        // prefill forward + 写 KV cache
        const size_t row_elems = nkvh * dh;
        for (size_t i = 0; i < L; ++i) {
            llaisysRmsNorm(norm1, x, m->weights.attn_norm_w[i], meta.epsilon);

            llaisysLinear(q_lin, norm1, m->weights.attn_q_w[i], m->weights.attn_q_b[i]);
            llaisysLinear(k_lin, norm1, m->weights.attn_k_w[i], m->weights.attn_k_b[i]);
            llaisysLinear(v_lin, norm1, m->weights.attn_v_w[i], m->weights.attn_v_b[i]);

            llaisysROPE(q, q, pos_ids, meta.theta);
            llaisysROPE(k, k, pos_ids, meta.theta);

            // 写入 cache（CPU memcpy）
            if (tensorGetDeviceType(m->k_cache[i]) == LLAISYS_DEVICE_CPU) {
                std::memcpy(tensorGetData(m->k_cache[i]), tensorGetData(k), ntoken * row_elems * elem_bytes);
                std::memcpy(tensorGetData(m->v_cache[i]), tensorGetData(v), ntoken * row_elems * elem_bytes);
            }

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

        // logits for last token
        llaisysTensor_t last = tensorSlice(x, 0, ntoken - 1, ntoken); // [1,hs]
        llaisysTensor_t logits = make_2d(dt, dev, dev_id, 1, voc);
        llaisysLinear(logits, last, m->weights.out_embed, nullptr);

        size_t logits_shape[1] = {voc};
        llaisysTensor_t logits_1d = tensorView(logits, logits_shape, 1);
        llaisysTensor_t max_idx = make_1d(LLAISYS_DTYPE_I64, dev, dev_id, 1);
        llaisysTensor_t max_val = make_1d(dt, dev, dev_id, 1);
        llaisysArgmax(max_idx, max_val, logits_1d);

        int64_t next_token = *reinterpret_cast<int64_t*>(tensorGetData(max_idx));

        // cleanup prefill temps
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

        m->seqlen = ntoken;
        return next_token;
    }

    // ============================
    // 2) Decode：ntoken == seqlen + 1，只算最后 1 个 token
    // ============================
    const size_t pos = ntoken - 1;
    const int64_t tok = token_ids[pos];
    const int64_t pos64 = static_cast<int64_t>(pos);

    tensorLoad(m->idx1, &tok);
    tensorLoad(m->pos1, &pos64);

    // x = embedding(last_token)
    llaisysEmbedding(m->x1, m->idx1, m->weights.in_embed);
    llaisysTensor_t x = m->x1;

    const size_t row_elems = nkvh * dh;

    for (size_t i = 0; i < L; ++i) {
        llaisysRmsNorm(m->norm1, x, m->weights.attn_norm_w[i], meta.epsilon);

        llaisysLinear(m->q_lin, m->norm1, m->weights.attn_q_w[i], m->weights.attn_q_b[i]);
        llaisysLinear(m->k_lin, m->norm1, m->weights.attn_k_w[i], m->weights.attn_k_b[i]);
        llaisysLinear(m->v_lin, m->norm1, m->weights.attn_v_w[i], m->weights.attn_v_b[i]);

        // rope on this step (qlen=1)
        llaisysROPE(m->q, m->q, m->pos1, meta.theta);
        llaisysROPE(m->k_step, m->k_step, m->pos1, meta.theta);

        // store new K/V into cache at position pos
        if (tensorGetDeviceType(m->k_cache[i]) == LLAISYS_DEVICE_CPU) {
            kv_cache_store_cpu(m->k_cache[i], pos, tensorGetData(m->k_step), row_elems, elem_bytes);
            kv_cache_store_cpu(m->v_cache[i], pos, tensorGetData(m->v_step), row_elems, elem_bytes);
        } else {
            // 先不支持非 CPU 的 KV cache（要么写 tensorCopy op，要么搞 device memcpy）
            return meta.end_token;
        }

        // view history [kvlen, nkvh, dh]
        const size_t kvlen = pos + 1;
        // view 会检查 numel 相等，所以这里必须用 slice 取前 kvlen 行
        llaisysTensor_t k_hist = tensorSlice(m->k_cache[i], 0, 0, kvlen); // [kvlen, nkvh, dh]
        llaisysTensor_t v_hist = tensorSlice(m->v_cache[i], 0, 0, kvlen); // [kvlen, nkvh, dh]


        // attention(q=1, kv=kvlen)
        llaisysSelfAttention(m->attn, m->q, k_hist, v_hist, scale);

        tensorDestroy(v_hist);
        tensorDestroy(k_hist);

        // proj + residual
        llaisysLinear(m->attn_proj, m->attn2d, m->weights.attn_o_w[i], nullptr);
        llaisysAdd(x, x, m->attn_proj);

        // mlp
        llaisysRmsNorm(m->norm2, x, m->weights.mlp_norm_w[i], meta.epsilon);
        llaisysLinear(m->gate, m->norm2, m->weights.mlp_gate_w[i], nullptr);
        llaisysLinear(m->up,   m->norm2, m->weights.mlp_up_w[i], nullptr);
        llaisysSwiGLU(m->swiglu, m->gate, m->up);
        llaisysLinear(m->down, m->swiglu, m->weights.mlp_down_w[i], nullptr);
        llaisysAdd(x, x, m->down);
    }

    llaisysRmsNorm(x, x, m->weights.out_norm_w, meta.epsilon);

    // logits + argmax
    llaisysLinear(m->logits, x, m->weights.out_embed, nullptr);
    llaisysArgmax(m->max_idx, m->max_val, m->logits_1d);

    int64_t next_token = *reinterpret_cast<int64_t*>(tensorGetData(m->max_idx));
    m->seqlen = ntoken;
    return next_token;
}
