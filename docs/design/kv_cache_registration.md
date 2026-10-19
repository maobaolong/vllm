# KV cache axis metadata at connector registration

KV connectors can override `register_kv_caches_with_layouts(kv_caches, layouts)`
to receive producer-declared axis semantics alongside the actual tensors.
The default implementation validates supplied layouts and delegates to the
existing `register_kv_caches(kv_caches)` hook. Existing connectors do not need
to change their method signatures. `MultiConnector` forwards the descriptors
to each child.

## Descriptor contract

`KVCacheTensorLayout.axes` is a tuple of `KVCacheAxis` enum members in **tensor
indexing order**, not physical stride order. `KVCacheLayout` continues to
describe the allocator's physical stride permutation. Consumers must use the
actual tensor's shape, strides, dtype, and storage offset for addressing.

Both GPU model runners pass descriptors for the standardized allocator's
logical `[block, head, token/state, content]` views, including views with
padded blocks or layers sharing an allocation. `CONTENT` is opaque: the
layer's `KVCacheSpec` describes its encoding and packing. Axis metadata alone
does not identify quantization scales or split packed K and V.

The mapping uses the same layer names as `kv_caches`. A missing entry means
unspecified semantics, not a default layout. Raw storage without semantic
layer views is omitted. A connector that requires explicit semantics must
reject such entries or use its existing spec-aware registration path.

## Custom allocators

An out-of-tree backend or model runner should construct descriptors where it
chooses the registered tensor's layout. It must describe the exact view
passed to the connector, rather than a later backend-private view.

For example, two RBLN-style layouts can have the identical shape
`[2, 2, heads, 1, tokens, head_dim]`. Shape or contiguous strides cannot
distinguish their block and K/V axes. Their declarations are different:

```python
from vllm.v1.kv_cache_layout import KVCacheAxis as Axis
from vllm.v1.kv_cache_layout import KVCacheTensorLayout

tail = (Axis.HEAD, Axis.SINGLETON, Axis.TOKEN, Axis.HEAD_DIM)
blocks_first = KVCacheTensorLayout((Axis.BLOCK, Axis.KV, *tail))
kv_first = KVCacheTensorLayout((Axis.KV, Axis.BLOCK, *tail))

connector.register_kv_caches_with_layouts(
    {"layer": cache}, {"layer": blocks_first}
)
```

The consumer can call `layout.axis_index(Axis.BLOCK)` to locate the block
dimension and `layout.validate_shape(cache.shape)` to check the rank and
fixed-size axes. Overriding connectors should validate descriptors before
using them. They can map the descriptor to an internal transfer format once
at registration, without importing a cache library's format enum into vLLM.

This hook does not itself update external connectors or hardware plugins.
They must opt in to produce and consume the metadata. The descriptor is an
in-process Python API; this change does not introduce a wire protocol.
