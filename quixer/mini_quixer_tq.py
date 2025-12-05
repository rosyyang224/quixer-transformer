# quixer/mini_quixer_tq.py - GPU优化版本(直接修改)

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


# -----------------------------
# TorchQuantum PQC 电路（优化版）
# -----------------------------
class TorchQuantumPQC(tq.QuantumModule):
    """
    GPU优化的PQC - 关键改进:
    1. 批量处理所有样本(而非逐个)
    2. 复用单个QuantumDevice
    3. 减少device创建开销
    """

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.measure_all = tq.MeasureAll(tq.PauliZ)

    def forward(self, angles: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        angles: [B, Q]
        params: [L, Q, 3]
        """
        B, Q = angles.shape
        L = params.shape[0]
        device = angles.device

        # ===== 关键优化: 创建单个batched device =====
        # 之前: 循环B次,每次创建device
        # 现在: 一次创建,bsz=B
        qdev = tq.QuantumDevice(n_wires=Q, bsz=B, device=device)

        # ===== 批量应用编码门 =====
        for q in range(Q):
            # angles[:, q] 是 [B] - 所有样本的第q个qubit的角度
            tqf.rx(qdev, wires=q, params=angles[:, q])

        # ===== 批量应用参数化层 =====
        for l in range(L):
            # 旋转层
            for q in range(Q):
                # params[l, q, 0] 是标量,需要repeat(B)变成[B]
                tqf.rx(qdev, wires=q, params=params[l, q, 0].expand(B))
                tqf.ry(qdev, wires=q, params=params[l, q, 1].expand(B))
                tqf.rz(qdev, wires=q, params=params[l, q, 2].expand(B))

            # CNOT 链
            for q in range(Q - 1):
                qdev.cnot(wires=[q, q + 1])

        # ===== 批量测量 =====
        z_exp = self.measure_all(qdev)  # [B, Q]

        return z_exp


# -----------------------------
# MiniQuixer 中间量子块
# -----------------------------
class MiniQuixerPQCBlock(nn.Module):
    """
    classical context [B, d_qkv] -> PQC -> [B, n_qubits]
    """

    def __init__(self, d_qkv: int, n_qubits: int, n_layers: int):
        super().__init__()
        self.d_qkv = d_qkv
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # context -> angles
        self.input_proj = nn.Linear(d_qkv, n_qubits)

        # PQC 可训练参数
        self.pqc_params = nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits, 3)
        )

        # TorchQuantum 电路(优化版)
        self.circuit = TorchQuantumPQC(n_qubits=n_qubits, n_layers=n_layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: [B, d_qkv]
        """
        angles = self.input_proj(context)           # [B, n_qubits]
        angles = torch.pi * torch.tanh(angles)      # 限制到 [-pi, pi]
        out = self.circuit(angles, self.pqc_params) # [B, n_qubits] - 批量处理!
        return out


# -----------------------------
# MiniQuixerTQ 主模型
# -----------------------------
class MiniQuixerTQ(nn.Module):
    """
    GPU优化版MiniQuixer
    """

    def __init__(
        self,
        n_qubits: int,
        n_tokens: int,
        qsvt_polynomial_degree: int,
        n_ansatz_layers: int,
        vocabulary_size: int,
        embedding_dimension: int,
        dropout: float,
        batch_size: int,
        device,
    ):
        super().__init__()
        self.device = device
        self.n_qubits = n_qubits
        self.n_tokens = n_tokens
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size

        # --- Token & Positional Embedding ---
        self.token_emb = nn.Embedding(vocabulary_size, embedding_dimension)
        self.pos_emb = nn.Embedding(n_tokens, embedding_dimension)
        self.dropout = nn.Dropout(dropout)

        # --- QKV projection ---
        self.d_qkv = embedding_dimension // 8
        self.qkv_proj = nn.Linear(embedding_dimension, 3 * self.d_qkv)

        # --- PQC Block (优化版) ---
        self.pqc_block = MiniQuixerPQCBlock(
            d_qkv=self.d_qkv,
            n_qubits=n_qubits,
            n_layers=qsvt_polynomial_degree,
        )

        # --- PQC 输出映射回 d_model ---
        self.map_back = nn.Linear(n_qubits, embedding_dimension)

        # --- FFN + LayerNorm ---
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dimension, 4 * embedding_dimension),
            nn.GELU(),
            nn.Linear(4 * embedding_dimension, embedding_dimension),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(embedding_dimension)
        self.ln2 = nn.LayerNorm(embedding_dimension)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T]  (T = window = n_tokens)
        """
        B, T = x.shape
        device = self.device

        # --- Embedding ---
        tok = self.token_emb(x)  # [B, T, d_model]
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos = self.pos_emb(pos_ids)

        h = tok + pos
        h = self.dropout(h)

        # --- QKV ---
        qkv = self.qkv_proj(h)  # [B, T, 3*d_qkv]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # --- 简单 context 聚合（mean-pool V） ---
        context = v.mean(dim=1)  # [B, d_qkv]

        # --- PQC Block (批量处理!) ---
        pqc_out = self.pqc_block(context)  # [B, n_qubits]
        pqc_expand = self.map_back(pqc_out).unsqueeze(1).expand(
            B, T, self.embedding_dimension
        )

        # --- Residual + LayerNorm ---
        h = self.ln1(h + pqc_expand)

        # --- FFN + Residual + LayerNorm ---
        h2 = self.ffn(h)
        h = self.ln2(h + h2)   # [B, T, d_model]

        # === 只用最后一个时间步预测下一个 token ===
        h_last = h[:, -1, :]   # [B, d_model]

        # 映射到 vocab 空间
        yhat = torch.matmul(h_last, self.token_emb.weight.t())  # [B, vocab_size]

        norm_avg = torch.tensor(0.0, device=device)

        return yhat, norm_avg