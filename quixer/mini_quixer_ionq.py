# quixer/mini_quixer_ionq.py

import torch
import torch.nn as nn
from ionq_backend import ionq_pqc_forward  # ← 这个会用到修改后的版本


class MiniQuixerPQCBlockIonQ(nn.Module):
    def __init__(self, input_proj: nn.Linear, pqc_params: torch.Tensor):
        super().__init__()
        self.input_proj = input_proj
        self.pqc_params = pqc_params

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            angles = self.input_proj(context)
            angles = torch.pi * torch.tanh(angles)


            # print(f"[MiniQuixerPQCBlockIonQ] Input context shape: {context.shape}")
            # print(f"[MiniQuixerPQCBlockIonQ] Angles shape: {angles.shape}")
            
            # 调用修改后的ionq_pqc_forward (会自动并行)
            out = ionq_pqc_forward(
                angles, 
                self.pqc_params,
                shots=1000,      # ← 添加: shots参数
                max_parallel=8   # ← 添加: 并行数
            )
        return out