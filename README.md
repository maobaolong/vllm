# vLLM KV-cache axis registration review guide

[Read the illustrated guide](https://maobaolong.github.io/vllm/).

This branch publishes a standalone reviewer guide for [vLLM #63699](https://github.com/vllm-project/vllm/pull/63699). It is separate from the implementation branch and is not part of the upstream code diff.

- Implementation snapshot: `383874621325a7057190711527ad268c183e0b19`.
- RBLN adoption snapshot: `a930b4daaf19ea083f3f7d8f80ed3fa82cc7a27d`.
- Status snapshot: 26 October 2026. Follow PR links for current status.
- Download `index.html` and open it in a browser for offline reading. All figures, styles and interactions are embedded. External links are source references.
- `assets/` contains screenshots for the PR description; the HTML itself does not depend on them.

The guide separates implemented producer/API work from proposed LMCache consumer and MP integration. It does not claim GPU, NPU, serving or LMCache E2E validation.

Prepared with OpenAI Codex assistance.
