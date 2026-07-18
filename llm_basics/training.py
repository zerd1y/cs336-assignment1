from __future__ import annotations

import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from torch import Tensor

from .experiment_tracking import ExperimentTracker, build_default_experiment_tracker


def compute_cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """根据未归一化 logits 计算平均交叉熵损失。

    Args:
        logits: 最后一维为类别 logits 的张量。
        targets: 整型目标张量，形状应与 ``logits.shape[:-1]`` 一致。

    Returns:
        表示平均交叉熵损失的标量张量。

    Raises:
        ValueError: 当 ``logits`` 没有类别维，或 ``targets`` 与 ``logits`` 形状不匹配时抛出。
    """
    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension.")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} must match logits shape "
            f"{tuple(logits.shape[:-1])} excluding the class dimension."
        )

    # 先减去每个位置上的最大 logit，避免指数运算时出现数值溢出。
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    logsumexp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    # 只提取目标类别对应的 logit，避免显式构造完整的 log_softmax。
    target_logits = torch.gather(shifted_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = logsumexp - target_logits
    return loss.mean()


class CustomAdamW(torch.optim.Optimizer):
    """仅使用原生 PyTorch 张量算子实现的 AdamW 优化器。"""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行一次优化器参数更新。

        Args:
            closure: 可选闭包，用于重新计算模型前向并返回损失。

        Returns:
            如果提供了 ``closure``，则返回其计算出的损失；否则返回 ``None``。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("CustomAdamW does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                state["t"] += 1
                t: int = state["t"]
                m: Tensor = state["m"]
                v: Tensor = state["v"]

                # 使用偏置修正后的等效步长，与标准 AdamW 更新形式保持一致。
                alpha_t = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
                if weight_decay != 0.0:
                    # AdamW 采用解耦权重衰减，直接对参数值做缩放。
                    param.data.mul_(1.0 - lr * weight_decay)

                # 一阶矩与二阶矩都用原地更新，减少额外张量分配。
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = torch.sqrt(v).add_(eps)
                param.data.addcdiv_(m, denom, value=-alpha_t)

        return loss


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机采样一批自回归语言模型训练样本。

    Args:
        data: 一维 NumPy 数组或 ``np.memmap``，内容为 token id。
        batch_size: 要采样的独立样本数。
        context_length: 每个输入序列与目标序列的长度。
        device: 返回张量所在的目标设备。

    Returns:
        二元组 ``(x, y)``，二者均为形状 ``(batch_size, context_length)`` 的
        ``torch.int64`` 张量，其中 ``y`` 相对 ``x`` 整体右移一个 token。

    Raises:
        ValueError: 当输入数据不是一维数组，或长度不足以构造训练样本时抛出。
    """
    if data.ndim != 1:
        raise ValueError(f"data must be one-dimensional, got shape {data.shape}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}.")

    max_start = data.shape[0] - context_length
    if max_start <= 0:
        raise ValueError(
            f"data length {data.shape[0]} is insufficient for context_length={context_length}."
        )

    # 每个起始位置都必须满足：x 长度为 context_length，y 还能再向后偏移 1 位。
    start_indices = np.random.randint(0, max_start, size=batch_size, dtype=np.int64)
    x_np = np.stack([data[start : start + context_length] for start in start_indices], axis=0)
    y_np = np.stack([data[start + 1 : start + context_length + 1] for start in start_indices], axis=0)

    # 先在 CPU 上整理成连续内存，再一次性转成 PyTorch 张量。
    x_cpu = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.int64))
    y_cpu = torch.from_numpy(np.ascontiguousarray(y_np, dtype=np.int64))

    device_obj = torch.device(device)
    if device_obj.type == "cpu":
        return x_cpu, y_cpu
    if device_obj.type == "cuda":
        # CUDA 路径下使用 pinned memory + non_blocking，减少主机到设备拷贝等待。
        x_cpu = x_cpu.pin_memory()
        y_cpu = y_cpu.pin_memory()
        return x_cpu.to(device_obj, non_blocking=True), y_cpu.to(device_obj, non_blocking=True)
    return x_cpu.to(device_obj), y_cpu.to(device_obj)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO,
) -> None:
    """序列化保存模型、优化器和训练步数状态。

    Args:
        model: 需要保存状态的模型。
        optimizer: 需要保存状态的优化器。
        iteration: 要持久化保存的当前训练步数。
        out: 输出路径或可写二进制文件对象。
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """从检查点恢复模型、优化器与训练步数状态。

    Args:
        src: 输入路径或可读二进制文件对象。
        model: 需要原地恢复状态的模型。
        optimizer: 需要原地恢复状态的优化器。

    Returns:
        检查点中保存的训练步数。

    Raises:
        KeyError: 当检查点中缺少必需字段时抛出。
        TypeError: 当序列化保存的 ``iteration`` 不是整数时抛出。
    """
    # 统一先映射到 CPU，避免在不同设备环境之间加载时报错。
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]
    if not isinstance(iteration, int):
        raise TypeError(f"Serialized iteration must be an int, got {type(iteration)!r}.")
    return iteration


def get_cosine_lr_with_warmup(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """返回“线性预热 + 余弦衰减”学习率调度在当前步的取值。

    Args:
        it: 当前训练迭代步。
        max_lr: 预热结束后达到的最大学习率。
        min_lr: 余弦衰减结束后的学习率下界。
        warmup_iters: 线性预热步数。
        cosine_cycle_iters: 余弦衰减结束时对应的步数。

    Returns:
        第 ``it`` 步对应的学习率数值。
    """
    if warmup_iters > 0 and it < warmup_iters:
        return (it / warmup_iters) * max_lr
    if it <= cosine_cycle_iters:
        # 防止 warmup_iters 与 cosine_cycle_iters 相等时出现除零。
        denominator = max(cosine_cycle_iters - warmup_iters, 1)
        cosine_ratio = (it - warmup_iters) / denominator
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * cosine_ratio)) * (max_lr - min_lr)
    return min_lr


@torch.no_grad()
def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> float:
    """按全局 L2 范数对梯度做原地裁剪。

    Args:
        parameters: 其梯度需要被裁剪的参数集合。
        max_norm: 允许的最大全局 L2 范数。
        eps: 防止除零的小常数。

    Returns:
        裁剪前的全局梯度范数，以 Python ``float`` 形式返回。
    """
    grads: list[Tensor] = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return 0.0

    # 使用 float32 累加梯度平方和，兼顾半精度训练下的数值稳定性。
    total_norm_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for grad in grads:
        grad_float = grad.detach().to(torch.float32)
        total_norm_sq.add_(torch.sum(grad_float * grad_float))

    global_norm = torch.sqrt(total_norm_sq).item()
    if global_norm > max_norm:
        scale = max_norm / (global_norm + eps)
        for grad in grads:
            grad.mul_(scale)

    return global_norm


@dataclass(slots=True)
class TrainingConfig:
    """自回归语言模型训练配置。

    Attributes:
        train_data_path: 训练集二进制 token 文件路径。
        val_data_path: 验证集二进制 token 文件路径。
        batch_size: 每个优化步骤采样的序列数量。
        context_length: 每个样本序列包含的 token 数。
        learning_rate: 学习率峰值。
        min_learning_rate: 余弦衰减结束后的最小学习率。
        warmup_steps: 线性预热步数。
        max_steps: 总训练步数。
        weight_decay: AdamW 的解耦权重衰减系数。
        grad_clip_norm: 全局梯度裁剪阈值。
        log_interval: 训练日志打印间隔。
        eval_interval: 验证集评估间隔。
        eval_batches: 每次评估时随机采样的验证 batch 数。
        checkpoint_interval: 检查点保存间隔。
        checkpoint_path: 检查点输出路径。
        resume_from: 可选的恢复训练检查点路径。
        device: 设备字符串，如 ``"cpu"``、``"cuda"`` 或 ``"mps"``。
        dataset_dtype: 二进制 token 文件使用的 NumPy 数据类型。
    """

    train_data_path: str | os.PathLike
    val_data_path: str | os.PathLike
    batch_size: int
    context_length: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    max_steps: int
    weight_decay: float
    grad_clip_norm: float
    log_interval: int
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int
    checkpoint_path: str | os.PathLike
    resume_from: str | os.PathLike | None = None
    device: str | torch.device = "cpu"
    dataset_dtype: np.dtype[Any] = np.dtype(np.uint16)
    experiment_tracker: ExperimentTracker | None = None


def load_token_memmap(path: str | os.PathLike, dtype: np.dtype[Any]) -> np.memmap:
    """以只读内存映射方式加载 token 数据集。

    Args:
        path: 扁平二进制 token 文件路径。
        dtype: 文件中单个元素的数据类型。

    Returns:
        只读的一维 ``np.memmap`` 对象。
    """
    return np.memmap(Path(path), dtype=dtype, mode="r")


@torch.no_grad()
def evaluate_language_model(
    model: torch.nn.Module,
    data: np.ndarray,
    config: TrainingConfig,
) -> tuple[float, float]:
    """在随机验证 batch 上估计平均损失与困惑度。

    Args:
        model: 返回形状为 ``(B, T, V)`` 的 logits 的语言模型。
        data: 验证集 token 数组。
        config: 控制 batch 采样方式的训练配置。

    Returns:
        二元组 ``(mean_loss, perplexity)``。
    """
    was_training = model.training
    model.eval()

    losses: list[float] = []
    for _ in range(config.eval_batches):
        x, y = get_batch(
            data=data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
        )
        logits = model(x)
        loss = compute_cross_entropy_loss(logits.view(-1, logits.shape[-1]), y.reshape(-1))
        # 这里只在记录标量时回到 CPU，避免在循环中引入额外同步开销。
        losses.append(float(loss.detach().cpu()))

    if was_training:
        model.train()

    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


def _set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """更新优化器中所有参数组的学习率。"""
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_language_model(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> int:
    """运行模块化的语言模型训练主循环。

    Args:
        model: 自回归语言模型，输出 logits 形状为 ``(B, T, V)``。
        config: 训练配置。

    Returns:
        训练完成后的最终步数。

    Raises:
        ValueError: 当调度参数或日志配置非法时抛出。
    """
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if config.log_interval <= 0:
        raise ValueError("log_interval must be positive.")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")
    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive.")
    if config.eval_batches <= 0:
        raise ValueError("eval_batches must be positive.")

    device = torch.device(config.device)
    tracker = config.experiment_tracker or build_default_experiment_tracker()
    model.to(device)
    model.train()

    # 训练集与验证集都使用 memmap 加载，适合大规模扁平 token 文件。
    train_data = load_token_memmap(config.train_data_path, config.dataset_dtype)
    val_data = load_token_memmap(config.val_data_path, config.dataset_dtype)

    optimizer = CustomAdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    start_iteration = 0
    if config.resume_from is not None:
        # 先恢复参数与优化器状态，再把模型显式迁移回目标设备。
        start_iteration = load_checkpoint(config.resume_from, model=model, optimizer=optimizer)
        model.to(device)

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for iteration in range(start_iteration, config.max_steps):
        # 每步显式刷新学习率，便于与自定义 warmup + cosine 调度保持一致。
        lr = get_cosine_lr_with_warmup(
            it=iteration,
            max_lr=config.learning_rate,
            min_lr=config.min_learning_rate,
            warmup_iters=config.warmup_steps,
            cosine_cycle_iters=max(config.max_steps - 1, 0),
        )
        _set_learning_rate(optimizer, lr)

        x, y = get_batch(
            data=train_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = compute_cross_entropy_loss(logits.view(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        # 先裁剪再更新，避免异常梯度破坏优化稳定性。
        grad_norm = clip_gradients(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        step = iteration + 1

        if step % config.log_interval == 0:
            train_loss = float(loss.detach().cpu())
            train_ppl = math.exp(train_loss)
            tracker.log_metrics(
                metrics={
                    "train/loss": train_loss,
                    "train/ppl": train_ppl,
                    "grad_norm": grad_norm,
                    "lr": lr,
                },
                gradient_step=step,
            )

        if step % config.eval_interval == 0:
            val_loss, val_ppl = evaluate_language_model(model=model, data=val_data, config=config)
            tracker.log_metrics(
                metrics={
                    "val/loss": val_loss,
                    "val/ppl": val_ppl,
                },
                gradient_step=step,
            )

        if step % config.checkpoint_interval == 0:
            save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=checkpoint_path)
            tracker.log_metrics(
                metrics={"checkpoint/save_count": 1.0},
                gradient_step=step,
            )

    save_checkpoint(model=model, optimizer=optimizer, iteration=config.max_steps, out=checkpoint_path)
    return config.max_steps


__all__ = [
    "CustomAdamW",
    "TrainingConfig",
    "clip_gradients",
    "compute_cross_entropy_loss",
    "evaluate_language_model",
    "get_batch",
    "get_cosine_lr_with_warmup",
    "load_checkpoint",
    "load_token_memmap",
    "save_checkpoint",
    "train_language_model",
]
