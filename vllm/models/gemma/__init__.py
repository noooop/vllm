# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING or current_platform.is_cuda():
    from .nvidia.model import GemmaForCausalLM
else:
    from vllm.model_executor.models.gemma import GemmaForCausalLM

__all__ = ["GemmaForCausalLM"]
