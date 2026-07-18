from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter


def _trunc_normal_parameter(
    shape: tuple[int, ...],
    *,
    mean: float,
    std: float,
    a: float,
    b: float,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> Parameter:
    """创建一个使用截断正态分布初始化的可训练参数。"""
    # 统一封装参数创建与初始化逻辑，避免各层重复样板代码。
    weight = torch.empty(shape, device=device, dtype=dtype)
    torch.nn.init.trunc_normal_(weight, mean=mean, std=std, a=a, b=b)
    return Parameter(weight)


class Linear(Module):
    """不带偏置的线性层，计算 ``y = x W^T``。"""

    weight: Parameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _trunc_normal_parameter(
            (out_features, in_features),
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        """对输入应用不带偏置的线性投影。"""
        # 权重形状为 (out_features, in_features)，因此这里实现的是 y = x W^T。
        return torch.einsum("...i,oi->...o", x, self.weight)


class Embedding(Module):
    """Token 嵌入查找表。"""

    weight: Parameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = _trunc_normal_parameter(
            (num_embeddings, embedding_dim),
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """根据 token 下标取出对应的嵌入向量。"""
        return self.weight[token_ids]


class RMSNorm(Module):
    """带可学习缩放参数的 RMSNorm。"""

    weight: Parameter

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(torch.ones(d_model))

    @property
    def gain(self) -> Parameter:
        """将缩放参数暴露为 ``gain`` 别名，同时保持 ``weight`` 的 state dict 兼容性。"""
        return self.weight

    def forward(self, x: Tensor) -> Tensor:
        """在最后一维上执行 RMSNorm，并使用 float32 保证数值稳定性。"""
        # 按要求先转成 float32 计算均方根，最后再转回输入 dtype。
        x_float = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
        normalized = (x_float / rms).to(x.dtype)
        return normalized * self.weight.to(device=x.device, dtype=x.dtype)


class RotaryPositionalEmbedding(Module):
    """带有正余弦缓存的旋转位置编码（RoPE）。"""

    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got d_k={d_k}.")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # RoPE 按二维一组做旋转，因此只在偶数下标上构造频率。
        pair_indices = torch.arange(0, d_k, 2, dtype=torch.float32)
        inv_freq = theta ** (-pair_indices / d_k)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.einsum("s,f->sf", positions, inv_freq)

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """按照 token 位置对 Query 或 Key 向量施加旋转。"""
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected last dimension {self.d_k}, got {x.shape[-1]}.")

        # 把最后一维拆成 (..., d_k / 2, 2)，每对相邻通道共享一组旋转角。
        x_pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        cos = self.cos_cached[token_positions].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(device=x.device, dtype=x.dtype)
        while cos.ndim < x_even.ndim:
            # 将位置维广播到任意前置 batch/head 维度上。
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # 二维旋转：
        # [x0', x1'] = [x0 cos - x1 sin, x0 sin + x1 cos]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return rotated.reshape_as(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    """使用基础张量算子实现数值稳定版 softmax。"""
    # 先减去最大值，避免 exp 溢出。
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """计算带可选布尔掩码的缩放点积注意力。"""
    d_k = q.shape[-1]
    scale = 1.0 / math.sqrt(d_k)
    # 注意力分数形状为 (..., queries, keys)。
    logits = torch.einsum("...qd,...kd->...qk", q, k) * scale

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("Attention mask must be a boolean tensor.")
        # False 的位置填成极小值，使 softmax 后概率接近 0。
        neg_inf = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits = torch.where(mask, logits, neg_inf)

    probs = softmax(logits, dim=-1)
    return torch.einsum("...qk,...kd->...qd", probs, v)


class CausalMultiHeadSelfAttention(Module):
    """带 RoPE 和因果掩码的 Decoder-only 多头自注意力。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        """将模型维拆分为多个注意力头。"""
        # (..., seq, d_model) -> (..., num_heads, seq, head_dim)
        return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """将多个注意力头重新合并回模型维。"""
        # (..., num_heads, seq, head_dim) -> (..., seq, d_model)
        x = x.transpose(-3, -2).contiguous()
        return x.reshape(*x.shape[:-2], self.d_model)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """对一段隐藏状态序列应用因果自注意力。"""
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")

        if token_positions is None:
            # 默认使用 0, 1, 2, ... 的绝对位置，并广播到 batch 维。
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            token_positions = token_positions.view(*((1,) * (x.ndim - 2)), seq_len).expand(*x.shape[:-1])

        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        # LLaMA 风格仅对 Q/K 施加 RoPE，V 不做旋转。
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # 因果掩码保证第 t 个位置只能看见 [0, t]。
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        causal_mask = causal_mask.view(*((1,) * (q.ndim - 2)), seq_len, seq_len)

        attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        return self.output_proj(self._merge_heads(attn_output))


def silu(x: Tensor) -> Tensor:
    """仅用逐元素算子实现 SiLU。"""
    return x * torch.sigmoid(x)


class SwiGLU(Module):
    """LLaMA 风格的门控前馈网络 SwiGLU。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil(((8.0 * d_model) / 3.0) / 64.0) * 64

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """应用 SwiGLU 前馈变换。"""
        # SwiGLU: W2( SiLU(W1 x) * (W3 x) )
        gate = silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


class TransformerBlock(Module):
    """包含因果自注意力与 SwiGLU 的 Pre-Norm Transformer Block。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """应用一个 Pre-Norm Transformer Block。"""
        # Pre-Norm 残差结构更稳定：
        # y = x + Attn(RMSNorm(x))
        # z = y + FFN(RMSNorm(y))
        y = x + self.attn(self.ln1(x), token_positions=token_positions)
        return y + self.ffn(self.ln2(y))


class TransformerLM(Module):
    """类似 LLaMA 的 Decoder-only Transformer 语言模型。"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        """输入一批 token 下标，返回对应位置的下一词预测 logits。"""
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(f"Sequence length {seq_len} exceeds context_length={self.context_length}.")

        # 先做 token embedding，再串联若干个 decoder block。
        x = self.token_embeddings(token_ids)
        token_positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        token_positions = token_positions.view(*((1,) * (token_ids.ndim - 1)), seq_len).expand_as(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # 最终 RMSNorm 后接 lm_head，输出未归一化 logits。
        x = self.ln_final(x)
        return self.lm_head(x)


def build_transformer_lm_from_state_dict(
    *,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
) -> TransformerLM:
    """实例化一个 ``TransformerLM`` 并加载外部 state dict。"""
    # 这个辅助函数主要用于测试或对拍外部参考权重。
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    model.load_state_dict(weights)
    return model


def run_linear(
    d_in: int,
    d_out: int,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    """使用外部权重运行 ``Linear`` 的便捷封装。"""
    layer = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    layer.weight.data.copy_(weights)
    return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Tensor,
    token_ids: Tensor,
) -> Tensor:
    """使用外部权重运行 ``Embedding`` 的便捷封装。"""
    layer = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    layer.weight.data.copy_(weights)
    return layer(token_ids)


def run_rmsnorm(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    """使用外部权重运行 ``RMSNorm`` 的便捷封装。"""
    layer = RMSNorm(d_model, eps=eps)
    layer.weight.data.copy_(weights.to(layer.weight.device, layer.weight.dtype))
    return layer(in_features)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """使用外部权重运行 ``SwiGLU`` 的便捷封装。"""
    layer = SwiGLU(d_model=d_model, d_ff=d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    layer.w1.weight.data.copy_(w1_weight)
    layer.w2.weight.data.copy_(w2_weight)
    layer.w3.weight.data.copy_(w3_weight)
    return layer(in_features)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
) -> Tensor:
    """对输入张量应用 RoPE 的便捷封装。"""
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """运行因果多头自注意力的便捷封装，默认使用顺序位置。"""
    attn = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=in_features.shape[-2],
        device=q_proj_weight.device,
        dtype=q_proj_weight.dtype,
    )
    attn.q_proj.weight.data.copy_(q_proj_weight)
    attn.k_proj.weight.data.copy_(k_proj_weight)
    attn.v_proj.weight.data.copy_(v_proj_weight)
    attn.output_proj.weight.data.copy_(o_proj_weight)
    positions = torch.arange(in_features.shape[-2], device=in_features.device, dtype=torch.long)
    positions = positions.view(*((1,) * (in_features.ndim - 2)), in_features.shape[-2]).expand(*in_features.shape[:-1])
    return attn(in_features, token_positions=positions)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor | None = None,
) -> Tensor:
    """运行带显式 RoPE 位置输入的因果多头自注意力便捷封装。"""
    attn = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
        device=q_proj_weight.device,
        dtype=q_proj_weight.dtype,
    )
    attn.q_proj.weight.data.copy_(q_proj_weight)
    attn.k_proj.weight.data.copy_(k_proj_weight)
    attn.v_proj.weight.data.copy_(v_proj_weight)
    attn.output_proj.weight.data.copy_(o_proj_weight)
    return attn(in_features, token_positions=token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    """使用外部权重运行单个 Transformer Block 的便捷封装。"""
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    block.load_state_dict(weights)
    positions = torch.arange(in_features.shape[-2], device=in_features.device, dtype=torch.long)
    positions = positions.view(1, in_features.shape[-2]).expand(in_features.shape[0], -1)
    return block(in_features, token_positions=positions)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    """使用外部权重运行完整 Transformer 语言模型的便捷封装。"""
    model = build_transformer_lm_from_state_dict(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        weights=weights,
    )
    return model(in_indices)


__all__ = [
    "CausalMultiHeadSelfAttention",
    "Embedding",
    "Linear",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "TransformerBlock",
    "TransformerLM",
    "build_transformer_lm_from_state_dict",
    "run_embedding",
    "run_linear",
    "run_multihead_self_attention",
    "run_multihead_self_attention_with_rope",
    "run_rmsnorm",
    "run_rope",
    "run_swiglu",
    "run_transformer_block",
    "run_transformer_lm",
    "scaled_dot_product_attention",
    "silu",
    "softmax",
]
