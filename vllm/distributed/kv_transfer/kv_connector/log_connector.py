# SPDX-License-Identifier: Apache-2.0
"""
Log KV Cache Connector for Distributed Machine Learning Inference

The LogConnector is used for testing purposes only.
"""

from typing import TYPE_CHECKING, List, Tuple, Union

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class LogConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):

        self.transfer_config = config.kv_transfer_config
        self.vllm_config = config

        logger.info("Initializing Log connector under kv_transfer_config %s",
                    self.transfer_config)

        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        logger.info("recv_kv_caches_and_hidden_states model_input %s", model_input)
        return None, False, model_input

    def recv_kv_caches_and_hidden_states_v1(
            self,
            model_executable: torch.nn.Module,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            inputs_embeds: torch.Tensor,
            scheduler_output: "SchedulerOutput",
            attn_metadata: "FlashAttentionMetadata",
            kv_caches: List[torch.Tensor],
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
    "SchedulerOutput"]:
        logger.info("recv_kv_caches_and_hidden_states_v1 scheduler_output %s, attn_metadata %s, inputs_embeds %s, positions %s, input_ids %s",
                    scheduler_output, attn_metadata, inputs_embeds, positions, input_ids)
        return None, False, input_ids, positions, inputs_embeds

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        logger.info("send_kv_caches_and_hidden_states model_input %s, hidden_or_intermediate_states %s",
                    model_input, hidden_or_intermediate_states)

    def send_kv_caches_and_hidden_states_v1(
        self,
        model_executable: torch.nn.Module,
        input_ids: torch.Tensor,
        scheduler_output: "SchedulerOutput",
        attn_metadata: "FlashAttentionMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        logger.info("send_kv_caches_and_hidden_states_v1 scheduler_output %s, attn_metadata %s, input_ids %s, hidden_or_intermediate_states %s",
                    scheduler_output, attn_metadata, input_ids, hidden_or_intermediate_states)

    def close(self):
        pass
