from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType, llaisysDataType_t, llaisysDeviceType_t

from pathlib import Path
import safetensors
import json
import ml_dtypes
import numpy as np
from array import array
from ctypes import (
    Structure,
    POINTER,
    c_size_t,
    c_float,
    c_int64,
    c_int,
    c_void_p,
    cast,
    byref,
)


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", c_void_p),
        ("out_embed", c_void_p),
        ("out_norm_w", c_void_p),
        ("attn_norm_w", POINTER(c_void_p)),
        ("attn_q_w", POINTER(c_void_p)),
        ("attn_q_b", POINTER(c_void_p)),
        ("attn_k_w", POINTER(c_void_p)),
        ("attn_k_b", POINTER(c_void_p)),
        ("attn_v_w", POINTER(c_void_p)),
        ("attn_v_b", POINTER(c_void_p)),
        ("attn_o_w", POINTER(c_void_p)),
        ("mlp_norm_w", POINTER(c_void_p)),
        ("mlp_gate_w", POINTER(c_void_p)),
        ("mlp_up_w", POINTER(c_void_p)),
        ("mlp_down_w", POINTER(c_void_p)),
    ]


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.device = device
        self.model_path = Path(model_path)
        self.weights = {}
        self._model = None
        self._meta = None
        self._weights_loaded = False

        for file in sorted(self.model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="numpy", device="cpu") as data_:
                for name_ in data_.keys():
                    self.weights[name_] = data_.get_tensor(name_)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        print("[qwen2] enter generate", flush=True)
        print("[qwen2] model is None?", self._model is None, flush=True)

        if max_new_tokens is None:
            max_new_tokens = 0

        if self._model is None:
            self._init_model()

        if not inputs:
            return []

        token_ids = array("q", inputs)
        end_token = int(self._meta.end_token)
        infer = LIB_LLAISYS.llaisysQwen2ModelInfer
        for _ in range(max_new_tokens):
            ptr = cast(token_ids.buffer_info()[0], POINTER(c_int64))
            print("[dbg] before infer ntoken=", len(token_ids), flush=True)
            next_token = int(infer(self._model, ptr, c_size_t(len(token_ids))))
            print("[dbg] after infer next_token=", next_token, flush=True)
            token_ids.append(next_token)
            if next_token == end_token:
                break

        return list(token_ids)

    def _init_model(self):
        print("[qwen2] _init_model: start", flush=True)
        self._load_c_api()
        print("[qwen2] _init_model: after _load_c_api", flush=True)

        config_path = self.model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config.json in {self.model_path}")

        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        def cfg_get(*keys, default=None):
            for key in keys:
                if key in cfg:
                    return cfg[key]
            return default

        torch_dtype = cfg_get("torch_dtype", "dtype", default=None)
        dtype = DataType.F32  # DEBUG: force fp32 to avoid bf16 kernels on CPU
        print("[dbg] FORCE dtype =", dtype, flush=True)

        nlayer = int(cfg_get("num_hidden_layers", "n_layer", "num_layers"))
        hs = int(cfg_get("hidden_size", "n_embd"))
        nh = int(cfg_get("num_attention_heads", "n_head"))
        nkvh = int(cfg_get("num_key_value_heads", "num_kv_heads", default=nh))
        dh = int(cfg_get("head_dim", default=hs // nh))
        di = int(cfg_get("intermediate_size", "n_inner"))
        maxseq = int(cfg_get("max_position_embeddings", "max_seq_len"))
        voc = int(cfg_get("vocab_size"))
        epsilon = float(cfg_get("rms_norm_eps", "layer_norm_epsilon", default=1e-5))
        theta = float(cfg_get("rope_theta", default=10000.0))
        end_token = cfg_get("eos_token_id", "end_token_id", default=None)
        if isinstance(end_token, list):
            end_token = end_token[0]
        if end_token is None:
            end_token = cfg_get("eos_token", default=0)
        end_token = int(end_token)

        self._meta = LlaisysQwen2Meta(
            llaisysDataType_t(dtype),
            c_size_t(nlayer),
            c_size_t(hs),
            c_size_t(nh),
            c_size_t(nkvh),
            c_size_t(dh),
            c_size_t(di),
            c_size_t(maxseq),
            c_size_t(voc),
            c_float(epsilon),
            c_float(theta),
            c_int64(end_token),
        )

        print("[dbg] meta.dtype =", int(self._meta.dtype), flush=True)

        LIB_LLAISYS.llaisysSetContextRuntime(
            llaisysDeviceType_t(self.device), c_int(0)
        )
        print("[qwen2] _init_model: meta nlayer/hs/nh/voc/maxseq =", nlayer, hs, nh, voc, maxseq, flush=True)
        print("[qwen2] _init_model: before ModelCreate", flush=True)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self._meta), llaisysDeviceType_t(self.device), None, c_int(0)
        )
        print("[qwen2] _init_model: after  ModelCreate, model ptr =", hex(int(self._model) if self._model else 0), flush=True)
        
        
        print("[qwen2] _init_model: before _load_weights", flush=True)
        self._load_weights(nlayer)
        print("[qwen2] _init_model: after  _load_weights", flush=True)

    def _load_c_api(self):
        if getattr(LIB_LLAISYS, "_qwen2_api_loaded", False):
            return

        LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [
            POINTER(LlaisysQwen2Meta),
            llaisysDeviceType_t,
            POINTER(c_int),
            c_int,
        ]
        LIB_LLAISYS.llaisysQwen2ModelCreate.restype = c_void_p

        LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
        LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

        LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [c_void_p]
        LIB_LLAISYS.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

        LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [
            c_void_p,
            POINTER(c_int64),
            c_size_t,
        ]
        LIB_LLAISYS.llaisysQwen2ModelInfer.restype = c_int64

        LIB_LLAISYS._qwen2_api_loaded = True

        LIB_LLAISYS.tensorLoad.argtypes = [c_void_p, c_void_p]
        LIB_LLAISYS.tensorLoad.restype = None

    def _map_dtype(self, torch_dtype):
        if torch_dtype is None and self.weights:
            torch_dtype = str(next(iter(self.weights.values())).dtype)
        if torch_dtype is None:
            return DataType.F32
        name = str(torch_dtype).lower()
        if "bfloat16" in name or "bf16" in name:
            return DataType.BF16
        if "float16" in name or "f16" in name:
            return DataType.F16
        if "float32" in name or "f32" in name:
            return DataType.F32
        if "float64" in name or "f64" in name:
            return DataType.F64
        return DataType.F32

    def _load_weights(self, nlayer: int):
        if self._weights_loaded:
            return

        weights = self.weights
        if not weights:
            raise ValueError("No safetensors weights found.")
        
        print("[qwen2] _load_weights: start", flush=True)
        w_struct = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        print("[qwen2] _load_weights: got weights struct", flush=True)
        print("[dbg] in_embed handle =", int(w_struct.in_embed or 0), flush=True)
        print("[dbg] out_embed handle =", int(w_struct.out_embed or 0), flush=True)
        print("[dbg] out_norm handle =", int(w_struct.out_norm_w or 0), flush=True)

        print("[dbg] attn_norm_w ptr =", w_struct.attn_norm_w, bool(w_struct.attn_norm_w), flush=True)
        if bool(w_struct.attn_norm_w):
            print("[dbg] attn_norm_w[0] =", int(w_struct.attn_norm_w[0] or 0), flush=True)
            print("[dbg] attn_norm_w[1] =", int(w_struct.attn_norm_w[1] or 0), flush=True)


        for i in range(nlayer):
            if i < 2:  # 只打印前两层，避免刷屏
                print(f"[qwen2] loading layer {i}", flush=True)

        def load_tensor(dst, src, name=""):
            if src is None:
                return
            if not dst:
                raise RuntimeError(f"dst tensor handle is NULL for {name}")

            # 1) cast to fp32 for CPU debug (avoid bf16/fp16 kernels)
            if src.dtype != np.float32:
                src = src.astype(np.float32)

            # 2) contiguous
            if not src.flags["C_CONTIGUOUS"]:
                src = np.ascontiguousarray(src)

            # 3) call C API
            print("[dbg] load", name, "src.dtype=", src.dtype, "shape=", src.shape, flush=True)
            LIB_LLAISYS.tensorLoad(dst, c_void_p(src.ctypes.data))


        in_embed = weights.get("model.embed_tokens.weight")
        if in_embed is None:
            raise KeyError("Missing model.embed_tokens.weight")
        out_embed = weights.get("lm_head.weight", in_embed)
        out_norm = weights.get("model.norm.weight")
        if out_norm is None:
            raise KeyError("Missing model.norm.weight")

        load_tensor(w_struct.in_embed, in_embed, "model.embed_tokens.weight")
        load_tensor(w_struct.out_embed, out_embed, "lm_head.weight")
        load_tensor(w_struct.out_norm_w, out_norm , "model.norm.weight")

        for i in range(nlayer):
            prefix = f"model.layers.{i}."
            attn_norm = weights.get(prefix + "input_layernorm.weight")
            if attn_norm is None:
                raise KeyError(f"Missing {prefix}input_layernorm.weight")
            mlp_norm = weights.get(prefix + "post_attention_layernorm.weight")
            if mlp_norm is None:
                raise KeyError(f"Missing {prefix}post_attention_layernorm.weight")

            q_w = weights.get(prefix + "self_attn.q_proj.weight")
            k_w = weights.get(prefix + "self_attn.k_proj.weight")
            v_w = weights.get(prefix + "self_attn.v_proj.weight")
            o_w = weights.get(prefix + "self_attn.o_proj.weight")
            if q_w is None or k_w is None or v_w is None or o_w is None:
                raise KeyError(f"Missing attention weights in {prefix}self_attn.*")

            q_b = weights.get(prefix + "self_attn.q_proj.bias")
            k_b = weights.get(prefix + "self_attn.k_proj.bias")
            v_b = weights.get(prefix + "self_attn.v_proj.bias")

            gate_w = weights.get(prefix + "mlp.gate_proj.weight")
            up_w = weights.get(prefix + "mlp.up_proj.weight")
            down_w = weights.get(prefix + "mlp.down_proj.weight")
            if gate_w is None or up_w is None or down_w is None:
                raise KeyError(f"Missing MLP weights in {prefix}mlp.*")

            load_tensor(w_struct.attn_norm_w[i], attn_norm, f"{prefix}input_layernorm.weight")
            load_tensor(w_struct.attn_q_w[i], q_w, f"{prefix}self_attn.q_proj.weight")
            if q_b is not None:
                load_tensor(w_struct.attn_q_b[i], q_b, f"{prefix}self_attn.q_proj.bias")
            load_tensor(w_struct.attn_k_w[i], k_w, f"{prefix}self_attn.k_proj.weight")
            if k_b is not None:
                load_tensor(w_struct.attn_k_b[i], k_b, f"{prefix}self_attn.k_proj.bias")
            load_tensor(w_struct.attn_v_w[i], v_w, f"{prefix}self_attn.v_proj.weight")
            if v_b is not None:
                load_tensor(w_struct.attn_v_b[i], v_b, f"{prefix}self_attn.v_proj.bias")
            load_tensor(w_struct.attn_o_w[i], o_w, f"{prefix}self_attn.o_proj.weight")
            load_tensor(w_struct.mlp_norm_w[i], mlp_norm, f"{prefix}post_attention_layernorm.weight")
            load_tensor(w_struct.mlp_gate_w[i], gate_w, f"{prefix}mlp.gate_proj.weight")
            load_tensor(w_struct.mlp_up_w[i], up_w, f"{prefix}mlp.up_proj.weight")
            load_tensor(w_struct.mlp_down_w[i], down_w, f"{prefix}mlp.down_proj.weight")

        self._weights_loaded = True
        print("[qwen2] _load_weights: done", flush=True)
