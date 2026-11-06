# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import nn

from vllm.model_executor.utils import set_weight_attrs

class GemmaRMSNorm(nn.Module):
    """Gemma-style RMS normalization backed by flashinfer kernels.

    When ``residual`` is given, the fused add + norm runs in place and the
    updated ``(x, residual)`` pair is returned.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        set_weight_attrs(self.weight, {"dummy_weight_value": 0.0})
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

        if residual is None:
            return gemma_rmsnorm(x, self.weight, self.variance_epsilon)

        # gemma_fused_add_rmsnorm mutates x and residual in place.
        gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
        return x, residual