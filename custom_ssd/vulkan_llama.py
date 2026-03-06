from __future__ import annotations

import ctypes as C
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_DEFAULT_BACKEND_DIR = Path.home() / ".lmstudio" / "extensions" / "backends" / "llama.cpp-win-x86_64-vulkan-avx2-2.5.1"
_BACKEND_LOCK = threading.Lock()
_BACKEND_CACHE: dict[tuple[Path, bool], "LlamaCppBackend"] = {}
_LOG_CALLBACK = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)


class LlamaCppError(RuntimeError):
    pass


class ModelCompatibilityError(LlamaCppError):
    pass


class _ModelParams(C.Structure):
    _fields_ = [
        ("devices", C.c_void_p),
        ("tensor_buft_overrides", C.c_void_p),
        ("n_gpu_layers", C.c_int32),
        ("split_mode", C.c_int),
        ("main_gpu", C.c_int32),
        ("tensor_split", C.c_void_p),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
        ("kv_overrides", C.c_void_p),
        ("vocab_only", C.c_bool),
        ("use_mmap", C.c_bool),
        ("use_direct_io", C.c_bool),
        ("use_mlock", C.c_bool),
        ("check_tensors", C.c_bool),
        ("use_extra_bufts", C.c_bool),
        ("no_host", C.c_bool),
        ("no_alloc", C.c_bool),
    ]


class _ContextParams(C.Structure):
    _fields_ = [
        ("n_ctx", C.c_uint32),
        ("n_batch", C.c_uint32),
        ("n_ubatch", C.c_uint32),
        ("n_seq_max", C.c_uint32),
        ("n_threads", C.c_int32),
        ("n_threads_batch", C.c_int32),
        ("rope_scaling_type", C.c_int),
        ("pooling_type", C.c_int),
        ("attention_type", C.c_int),
        ("flash_attn_type", C.c_int),
        ("rope_freq_base", C.c_float),
        ("rope_freq_scale", C.c_float),
        ("yarn_ext_factor", C.c_float),
        ("yarn_attn_factor", C.c_float),
        ("yarn_beta_fast", C.c_float),
        ("yarn_beta_slow", C.c_float),
        ("yarn_orig_ctx", C.c_uint32),
        ("defrag_thold", C.c_float),
        ("cb_eval", C.c_void_p),
        ("cb_eval_user_data", C.c_void_p),
        ("type_k", C.c_int),
        ("type_v", C.c_int),
        ("abort_callback", C.c_void_p),
        ("abort_callback_data", C.c_void_p),
        ("embeddings", C.c_bool),
        ("offload_kqv", C.c_bool),
        ("no_perf", C.c_bool),
        ("op_offload", C.c_bool),
        ("swa_full", C.c_bool),
        ("kv_unified", C.c_bool),
        ("samplers", C.c_void_p),
        ("n_samplers", C.c_size_t),
    ]


class _Batch(C.Structure):
    _fields_ = [
        ("n_tokens", C.c_int32),
        ("token", C.POINTER(C.c_int32)),
        ("embd", C.POINTER(C.c_float)),
        ("pos", C.POINTER(C.c_int32)),
        ("n_seq_id", C.POINTER(C.c_int32)),
        ("seq_id", C.POINTER(C.POINTER(C.c_int32))),
        ("logits", C.POINTER(C.c_int8)),
    ]


@dataclass(slots=True)
class ContextSnapshot:
    state: bytes
    n_past: int
    last_logits_index: int | None


@dataclass(slots=True)
class ContextConfig:
    n_ctx: int = 2048
    n_batch: int = 256
    n_ubatch: int = 256
    n_seq_max: int = 1
    n_threads: int = 8
    n_threads_batch: int = 8
    offload_kqv: bool = True
    flash_attn_type: int = -1


def get_backend(*, backend_dir: Path | None = None, quiet: bool = False) -> "LlamaCppBackend":
    resolved = (backend_dir or _DEFAULT_BACKEND_DIR).resolve()
    key = (resolved, quiet)
    with _BACKEND_LOCK:
        backend = _BACKEND_CACHE.get(key)
        if backend is None:
            backend = LlamaCppBackend(resolved, quiet=quiet)
            _BACKEND_CACHE[key] = backend
    return backend


class LlamaCppBackend:
    def __init__(self, backend_dir: Path, *, quiet: bool = False) -> None:
        self.backend_dir = backend_dir
        self.quiet = quiet
        self._log_callback = None
        if not self.backend_dir.exists():
            raise FileNotFoundError(f"Vulkan backend directory does not exist: {self.backend_dir}")
        os.add_dll_directory(str(self.backend_dir))
        self.dll = C.CDLL(str(self.backend_dir / "llama.dll"))
        self._configure_exports()
        if self.quiet and hasattr(self.dll, "llama_log_set"):
            self._log_callback = _LOG_CALLBACK(lambda _level, _text, _user_data: None)
            self.dll.llama_log_set(self._log_callback, None)
        self.dll.llama_backend_init()

    def _configure_exports(self) -> None:
        dll = self.dll

        dll.llama_backend_init.argtypes = []
        dll.llama_backend_init.restype = None

        if hasattr(dll, "llama_log_set"):
            dll.llama_log_set.argtypes = [C.c_void_p, C.c_void_p]
            dll.llama_log_set.restype = None

        dll.llama_context_default_params.argtypes = []
        dll.llama_context_default_params.restype = _ContextParams

        dll.llama_load_model_from_file.argtypes = [C.c_char_p, _ModelParams]
        dll.llama_load_model_from_file.restype = C.c_void_p

        dll.llama_new_context_with_model.argtypes = [C.c_void_p, _ContextParams]
        dll.llama_new_context_with_model.restype = C.c_void_p

        dll.llama_model_get_vocab.argtypes = [C.c_void_p]
        dll.llama_model_get_vocab.restype = C.c_void_p

        dll.llama_vocab_n_tokens.argtypes = [C.c_void_p]
        dll.llama_vocab_n_tokens.restype = C.c_int32

        dll.llama_vocab_is_eog.argtypes = [C.c_void_p, C.c_int32]
        dll.llama_vocab_is_eog.restype = C.c_bool

        dll.llama_tokenize.argtypes = [
            C.c_void_p,
            C.c_char_p,
            C.c_int32,
            C.POINTER(C.c_int32),
            C.c_int32,
            C.c_bool,
            C.c_bool,
        ]
        dll.llama_tokenize.restype = C.c_int32

        dll.llama_detokenize.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_int32),
            C.c_int32,
            C.c_char_p,
            C.c_int32,
            C.c_bool,
            C.c_bool,
        ]
        dll.llama_detokenize.restype = C.c_int32

        dll.llama_decode.argtypes = [C.c_void_p, _Batch]
        dll.llama_decode.restype = C.c_int32

        dll.llama_sampler_init_greedy.argtypes = []
        dll.llama_sampler_init_greedy.restype = C.c_void_p

        dll.llama_sampler_sample.argtypes = [C.c_void_p, C.c_void_p, C.c_int32]
        dll.llama_sampler_sample.restype = C.c_int32

        dll.llama_sampler_free.argtypes = [C.c_void_p]
        dll.llama_sampler_free.restype = None

        dll.llama_get_logits_ith.argtypes = [C.c_void_p, C.c_int32]
        dll.llama_get_logits_ith.restype = C.POINTER(C.c_float)

        dll.llama_get_state_size.argtypes = [C.c_void_p]
        dll.llama_get_state_size.restype = C.c_size_t

        dll.llama_copy_state_data.argtypes = [C.c_void_p, C.POINTER(C.c_uint8)]
        dll.llama_copy_state_data.restype = C.c_size_t

        dll.llama_set_state_data.argtypes = [C.c_void_p, C.POINTER(C.c_uint8)]
        dll.llama_set_state_data.restype = C.c_size_t

        dll.llama_model_free.argtypes = [C.c_void_p]
        dll.llama_model_free.restype = None

        dll.llama_free.argtypes = [C.c_void_p]
        dll.llama_free.restype = None

    def load_model(
        self,
        model_path: str | Path,
        *,
        n_gpu_layers: int = -1,
        use_mmap: bool = True,
        use_extra_bufts: bool = True,
    ) -> "LlamaModel":
        path = Path(model_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model does not exist: {path}")
        params = _ModelParams()
        params.n_gpu_layers = n_gpu_layers
        params.use_mmap = use_mmap
        params.use_extra_bufts = use_extra_bufts
        handle = self.dll.llama_load_model_from_file(str(path).encode("utf-8"), params)
        if not handle:
            raise LlamaCppError(f"Failed to load model via Vulkan backend: {path}")
        return LlamaModel(self, path, handle, n_gpu_layers=n_gpu_layers)


class LlamaModel:
    def __init__(self, backend: LlamaCppBackend, path: Path, handle: int, *, n_gpu_layers: int) -> None:
        self.backend = backend
        self.path = path
        self.handle = handle
        self.n_gpu_layers = n_gpu_layers
        self.vocab = self.backend.dll.llama_model_get_vocab(self.handle)
        self.n_vocab = int(self.backend.dll.llama_vocab_n_tokens(self.vocab))
        self._closed = False

    def create_context(self, config: ContextConfig | None = None) -> "LlamaContext":
        cfg = config or ContextConfig()
        params = self.backend.dll.llama_context_default_params()
        params.n_ctx = cfg.n_ctx
        params.n_batch = cfg.n_batch
        params.n_ubatch = cfg.n_ubatch
        params.n_seq_max = cfg.n_seq_max
        params.n_threads = cfg.n_threads
        params.n_threads_batch = cfg.n_threads_batch
        params.offload_kqv = cfg.offload_kqv
        params.flash_attn_type = cfg.flash_attn_type
        handle = self.backend.dll.llama_new_context_with_model(self.handle, params)
        if not handle:
            raise LlamaCppError(f"Failed to create context for model: {self.path}")
        return LlamaContext(self, handle, cfg)

    def tokenize(self, text: str, *, add_special: bool = True, parse_special: bool = False) -> list[int]:
        encoded = text.encode("utf-8")
        needed = self.backend.dll.llama_tokenize(
            self.vocab,
            encoded,
            len(encoded),
            None,
            0,
            add_special,
            parse_special,
        )
        count = abs(int(needed))
        buffer = (C.c_int32 * count)()
        written = self.backend.dll.llama_tokenize(
            self.vocab,
            encoded,
            len(encoded),
            buffer,
            count,
            add_special,
            parse_special,
        )
        return [int(buffer[idx]) for idx in range(int(written))]

    def detokenize(self, tokens: list[int], *, remove_special: bool = False, unparse_special: bool = True) -> str:
        if not tokens:
            return ""
        token_array = (C.c_int32 * len(tokens))(*tokens)
        buf = C.create_string_buffer(max(64, len(tokens) * 16))
        needed = self.backend.dll.llama_detokenize(
            self.vocab,
            token_array,
            len(tokens),
            buf,
            len(buf),
            remove_special,
            unparse_special,
        )
        if needed < 0:
            buf = C.create_string_buffer(abs(int(needed)) + 8)
            needed = self.backend.dll.llama_detokenize(
                self.vocab,
                token_array,
                len(tokens),
                buf,
                len(buf),
                remove_special,
                unparse_special,
            )
        return buf.raw[: int(needed)].decode("utf-8", errors="ignore")

    def is_eog(self, token: int) -> bool:
        return bool(self.backend.dll.llama_vocab_is_eog(self.vocab, int(token)))

    def close(self) -> None:
        if not self._closed:
            self.backend.dll.llama_model_free(C.c_void_p(self.handle))
            self._closed = True

    def __enter__(self) -> "LlamaModel":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class LlamaContext:
    def __init__(self, model: LlamaModel, handle: int, config: ContextConfig) -> None:
        self.model = model
        self.backend = model.backend
        self.handle = handle
        self.config = config
        self.n_past = 0
        self._last_logits_index: int | None = None
        self._greedy_sampler = self.backend.dll.llama_sampler_init_greedy()
        self._closed = False

    def _make_batch(self, tokens: list[int], *, logits_all: bool) -> _Batch:
        n_tokens = len(tokens)
        token_array = (C.c_int32 * n_tokens)(*tokens)
        pos_array = (C.c_int32 * n_tokens)(*range(self.n_past, self.n_past + n_tokens))
        n_seq_array = (C.c_int32 * n_tokens)(*([1] * n_tokens))
        seq_values = [(C.c_int32 * 1)(0) for _ in range(n_tokens)]
        seq_ptrs = (C.POINTER(C.c_int32) * n_tokens)(
            *[C.cast(item, C.POINTER(C.c_int32)) for item in seq_values]
        )
        logits_flags = [1] * n_tokens if logits_all else ([0] * max(0, n_tokens - 1) + [1])
        logits_array = (C.c_int8 * n_tokens)(*logits_flags)
        batch = _Batch(
            n_tokens,
            token_array,
            None,
            pos_array,
            n_seq_array,
            seq_ptrs,
            logits_array,
        )
        batch._keepalive = (token_array, pos_array, n_seq_array, seq_values, seq_ptrs, logits_array)  # type: ignore[attr-defined]
        return batch

    def decode(self, tokens: list[int], *, logits_all: bool = False) -> None:
        if not tokens:
            return
        batch = self._make_batch(tokens, logits_all=logits_all)
        rc = self.backend.dll.llama_decode(C.c_void_p(self.handle), batch)
        if rc != 0:
            raise LlamaCppError(f"llama_decode failed with status {rc}")
        self.n_past += len(tokens)
        self._last_logits_index = len(tokens) - 1

    def sample_greedy(self) -> int:
        token = self.backend.dll.llama_sampler_sample(
            C.c_void_p(self._greedy_sampler),
            C.c_void_p(self.handle),
            C.c_int32(-1),
        )
        return int(token)

    def current_logits(self) -> np.ndarray:
        if self._last_logits_index is None:
            raise LlamaCppError("No logits are available yet on this context")
        return self.logits_for(self._last_logits_index)

    def logits_for(self, token_index: int) -> np.ndarray:
        ptr = self.backend.dll.llama_get_logits_ith(C.c_void_p(self.handle), int(token_index))
        if not ptr:
            raise LlamaCppError(f"Logits for token index {token_index} are not available")
        return np.ctypeslib.as_array(ptr, shape=(self.model.n_vocab,))

    def argmax_for(self, token_index: int) -> int:
        return int(self.logits_for(token_index).argmax())

    def top_tokens_for(self, token_index: int, k: int) -> list[int]:
        logits = self.logits_for(token_index)
        if k >= logits.shape[0]:
            order = np.argsort(logits)[::-1]
            return [int(item) for item in order.tolist()]
        idx = np.argpartition(logits, -k)[-k:]
        ordered = idx[np.argsort(logits[idx])[::-1]]
        return [int(item) for item in ordered.tolist()]

    def snapshot(self) -> ContextSnapshot:
        size = int(self.backend.dll.llama_get_state_size(C.c_void_p(self.handle)))
        buffer = (C.c_uint8 * size)()
        copied = int(self.backend.dll.llama_copy_state_data(C.c_void_p(self.handle), buffer))
        if copied <= 0:
            raise LlamaCppError("Failed to snapshot context state")
        return ContextSnapshot(bytes(buffer[:copied]), self.n_past, self._last_logits_index)

    def restore(self, snapshot: ContextSnapshot) -> None:
        buffer = (C.c_uint8 * len(snapshot.state)).from_buffer_copy(snapshot.state)
        restored = int(self.backend.dll.llama_set_state_data(C.c_void_p(self.handle), buffer))
        if restored != len(snapshot.state):
            raise LlamaCppError("Failed to restore full context state")
        self.n_past = snapshot.n_past
        self._last_logits_index = snapshot.last_logits_index

    def close(self) -> None:
        if not self._closed:
            self.backend.dll.llama_sampler_free(C.c_void_p(self._greedy_sampler))
            self.backend.dll.llama_free(C.c_void_p(self.handle))
            self._closed = True

    def __enter__(self) -> "LlamaContext":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
