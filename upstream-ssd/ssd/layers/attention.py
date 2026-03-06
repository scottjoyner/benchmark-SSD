import torch
from torch import nn
import torch.nn.functional as F
from ssd.compat import optional_import, require_cuda_backend
from ssd.utils.context import get_context

triton = optional_import("triton")
tl = optional_import("triton.language")
_sgl_flash_attn = optional_import("sgl_kernel.flash_attn")
if _sgl_flash_attn is not None:
    flash_attn_varlen_func = _sgl_flash_attn.flash_attn_varlen_func
    flash_attn_with_kvcache = _sgl_flash_attn.flash_attn_with_kvcache
else:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


if triton is not None and tl is not None:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot.to(tl.int64) * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)
else:
    store_kvcache_kernel = None


def _repeat_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.size(1) == num_heads:
        return x
    repeats = num_heads // x.size(1)
    return x.repeat_interleave(repeats, dim=1)


def _fallback_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    num_heads: int,
    is_causal: bool,
) -> torch.Tensor:
    q = q.transpose(0, 1).unsqueeze(0)
    k = _repeat_kv(k, num_heads).transpose(0, 1).unsqueeze(0)
    v = _repeat_kv(v, num_heads).transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    return out.squeeze(0).transpose(0, 1)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    require_cuda_backend(
        "Attention.store_kvcache",
        [
            name
            for name, module in {
                "triton": triton,
                "triton.language": tl,
            }.items()
            if module is None
        ],
    )
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        draft: bool = False,
        speculate: bool = False,
        draft_async: bool = False,
        use_eagle: bool = False,
        F: int = 1,
        K: int = 1,
    ):
        super().__init__()
        self.fallback_backend = any(
            module is None
            for module in (triton, tl, _sgl_flash_attn)
        )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.draft = draft
        self.speculate = speculate
        self.draft_async = draft_async
        self.use_eagle = use_eagle
        self.prefill_wrappers = {}
        self.F = F # async_fan_out
        self.K = K # speculate_k
        self.only_prefill_wrapper = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        k_cache, v_cache = self.k_cache, self.v_cache

        context = get_context()
        if self.fallback_backend:
            if self.k_cache.numel() and self.v_cache.numel() and context.slot_mapping is not None:
                try:
                    store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
                except Exception:
                    pass

            if context.is_prefill:
                o = _fallback_attention(q, k, v, self.scale, self.num_heads, is_causal=True)
            else:
                if self.k_cache.numel() and self.v_cache.numel():
                    k_src = self.k_cache.view(-1, self.num_kv_heads, self.head_dim)
                    v_src = self.v_cache.view(-1, self.num_kv_heads, self.head_dim)
                else:
                    k_src, v_src = k, v
                o = _fallback_attention(q, k_src, v_src, self.scale, self.num_heads, is_causal=False)
            return o.view(-1, self.num_heads * self.head_dim)

        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            k, v = k.view(-1, self.num_kv_heads, self.head_dim), v.view(-1, self.num_kv_heads, self.head_dim)
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True)
        else:
            # verify/glue decode: multi-query with cu_seqlens_q (K+1 or variable per seq)
            verify_or_glue = (
                self.speculate and context.cu_seqlens_q is not None
            )
            decode = not verify_or_glue
            tree_decode = (
                decode and self.speculate and self.draft and self.draft_async
                and not context.is_jit
            )

            if verify_or_glue:
                assert context.context_lens is not None
                o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                        cache_seqlens=context.context_lens, page_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True,
                                        cu_seqlens_q=context.cu_seqlens_q, max_seqlen_q=context.max_seqlen_q,
                                        )

            elif tree_decode:
                if self.only_prefill_wrapper is not None:
                    prefill_wrapper = self.only_prefill_wrapper
                else:
                    mq_len = self.F * (self.K+1)
                    bs = q.shape[0] // mq_len
                    wrapper_bs = None
                    for available_bs in sorted(self.prefill_wrappers.keys()):
                        if available_bs >= bs:
                            wrapper_bs = available_bs
                            break
                    prefill_wrapper = self.prefill_wrappers[wrapper_bs]
                o = prefill_wrapper.run(q, (self.k_cache, self.v_cache))
            else: # single query decode
                q = q.unsqueeze(1)
                o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                            cache_seqlens=context.context_lens, page_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True,
                                            )

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
