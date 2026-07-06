### Song Yuchen · 宋雨辰

MS @ Harbin Institute of Technology — **efficient LLM inference & systems**.
I build small, readable reference implementations of the tricks that make large models cheap to serve.

#### liteinfer
A pure-Python reference toolkit for efficient LLM inference — runs offline, no GPU required.

- **Quantization** — post-training INT8 / INT4 weight quantization
- **Paged KV-cache** — block-based cache management for long contexts
- **Speculative decoding** — draft-and-verify for faster generation
- **Reproducible** — NumPy core, optional PyTorch backend, CI green

→ [github.com/luke-ward88/liteinfer](https://github.com/luke-ward88/liteinfer)

<sub>Harbin Institute of Technology · LLM inference · quantization · paged KV-cache · speculative decoding</sub>
