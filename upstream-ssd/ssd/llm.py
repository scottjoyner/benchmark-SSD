class LLM:
    """Lazy wrapper so importing `ssd` does not immediately require the CUDA stack."""

    def __new__(cls, *args, **kwargs):
        from ssd.compat import optional_import, require_cuda_backend

        require_cuda_backend(
            "LLM",
            [
                name
                for name, module in {
                    "flashinfer": optional_import("flashinfer"),
                    "sgl_kernel.flash_attn": optional_import("sgl_kernel.flash_attn"),
                }.items()
                if module is None
            ],
        )
        from ssd.engine.llm_engine import LLMEngine

        return LLMEngine(*args, **kwargs)
