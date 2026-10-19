# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheAxis as Axis
from vllm.v1.kv_cache_layout import KVCacheTensorLayout

pytestmark = pytest.mark.cpu_test


def test_equal_sized_block_and_kv_axes_are_unambiguous():
    """Axis semantics must survive even when both dimensions have size two."""
    tail = (Axis.HEAD, Axis.SINGLETON, Axis.TOKEN, Axis.HEAD_DIM)
    blocks_first = KVCacheTensorLayout((Axis.BLOCK, Axis.KV, *tail))
    kv_first = KVCacheTensorLayout((Axis.KV, Axis.BLOCK, *tail))
    tensor = torch.arange(8).reshape(2, 2, 1, 1, 2, 1)

    blocks_first.validate_shape(tensor.shape)
    kv_first.validate_shape(tensor.shape)
    assert blocks_first.axis_index(Axis.BLOCK) == 0
    assert kv_first.axis_index(Axis.BLOCK) == 1
    assert not torch.equal(
        tensor.select(blocks_first.axis_index(Axis.BLOCK), 0),
        tensor.select(kv_first.axis_index(Axis.BLOCK), 0),
    )


@pytest.mark.parametrize(
    ("shape", "message"),
    [((3, 2, 1), "rank"), ((3, 3, 1, 4), "KV axis"), ((3, 2, 2, 4), "SINGLETON")],
)
def test_rejects_tensor_shape_inconsistent_with_declared_axes(shape, message):
    layout = KVCacheTensorLayout((Axis.BLOCK, Axis.KV, Axis.SINGLETON, Axis.HEAD_DIM))
    with pytest.raises(ValueError, match=message):
        layout.validate_shape(shape)


def test_rejects_duplicate_semantic_axes():
    with pytest.raises(ValueError, match="unique"):
        KVCacheTensorLayout((Axis.BLOCK, Axis.BLOCK))


def test_indexing_order_does_not_imply_contiguous_storage():
    tensor = torch.empty(2, 3, 4, 5).permute(0, 2, 1, 3)
    layout = KVCacheTensorLayout((Axis.BLOCK, Axis.HEAD, Axis.TOKEN, Axis.CONTENT))
    layout.validate_shape(tensor.shape)
    assert not tensor.is_contiguous()
    assert tensor.stride(layout.axis_index(Axis.TOKEN)) == 20


@pytest.mark.parametrize("uniform", [False, True])
def test_standard_allocator_descriptors_exclude_raw_and_unregistered_storage(uniform):
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float32
    )
    group_spec = (
        UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"layer": spec})
        if uniform
        else spec
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer"], group_spec),
            KVCacheGroupSpec(
                ["raw"], HiSparseResidentSpec(block_size=4, page_size=128)
            ),
            KVCacheGroupSpec(["not_registered"], spec),
        ],
    )

    layouts = config.get_tensor_layouts(["layer", "raw", "custom"])

    assert layouts == {
        "layer": KVCacheTensorLayout((Axis.BLOCK, Axis.HEAD, Axis.TOKEN, Axis.CONTENT))
    }
