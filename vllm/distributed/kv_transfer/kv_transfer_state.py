# SPDX-License-Identifier: Apache-2.0
import dataclasses
from typing import TYPE_CHECKING, Optional

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_KV_CONNECTOR_AGENT: Optional[KVConnectorBaseType] = None
_KV_OFFLOAD_CONNECTOR_AGENT: Optional[KVConnectorBaseType] = None

def get_kv_transfer_group() -> KVConnectorBaseType:
    assert _KV_CONNECTOR_AGENT is not None, (
        "disaggregated KV cache transfer parallel group is not initialized")
    return _KV_CONNECTOR_AGENT


def has_kv_transfer_group() -> bool:
    return _KV_CONNECTOR_AGENT is not None


def is_v1_kv_transfer_group(
        connector: Optional[KVConnectorBaseType] = None) -> bool:
    """Check if the KV connector is the v1 connector.
    If the argument is None, it will check the global KV connector

    Args:
        connector: The KV connector to check. If None, it will check the
            global KV connector.

    Note:
        This function will no-longer be needed after the v1 KV connector
        becomes the default.
    """
    if connector is None:
        connector = _KV_CONNECTOR_AGENT

    if connector is None:
        return False

    return isinstance(connector, KVConnectorBase_V1)


def get_kv_offload_group() -> KVConnectorBaseType:
    assert _KV_OFFLOAD_CONNECTOR_AGENT is not None, (
        "KV offload parallel group is not initialized")
    return _KV_OFFLOAD_CONNECTOR_AGENT


def has_kv_offload_group() -> bool:
    return _KV_OFFLOAD_CONNECTOR_AGENT is not None


def ensure_kv_transfer_initialized(vllm_config: "VllmConfig") -> None:
    """
    Initialize KV cache transfer parallel groups.
    """
    global _KV_CONNECTOR_AGENT, _KV_OFFLOAD_CONNECTOR_AGENT

    # Initialize original transfer group
    if (vllm_config.kv_transfer_config is not None and
        vllm_config.kv_transfer_config.is_kv_transfer_instance and
        _KV_CONNECTOR_AGENT is None):
        if envs.VLLM_USE_V1:
            _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector_v1(
                config=vllm_config, role=KVConnectorRole.WORKER)
        else:
            _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector_v0(
                rank=get_world_group().rank,
                local_rank=get_world_group().local_rank,
                config=vllm_config,
            )

    # Initialize offload group
    if (vllm_config.kv_offload_config is not None and
        vllm_config.kv_offload_config.kv_connector_extra_config.get("offload", False) and
        _KV_OFFLOAD_CONNECTOR_AGENT is None):
        logger.info("Initializing KV offload parallel group")
        # Create a config copy with kv_offload_config replacing kv_transfer_config
        offload_config = dataclasses.replace(
            vllm_config,
            kv_transfer_config=vllm_config.kv_offload_config
        )
        if envs.VLLM_USE_V1:
            _KV_OFFLOAD_CONNECTOR_AGENT = KVConnectorFactory.create_connector_v1(
                config=offload_config, role=KVConnectorRole.WORKER)
        else:
            _KV_OFFLOAD_CONNECTOR_AGENT = KVConnectorFactory.create_connector_v0(
                rank=get_world_group().rank,
                local_rank=get_world_group().local_rank,
                config=offload_config,
            )
