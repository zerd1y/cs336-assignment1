from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bpe import (
    GPT2_PRETOKEN_PATTERN,
    Tokenizer,
    _WordEntry,
    _count_pairs,
    _dedupe_nonempty,
    _merge_symbols,
    _token_bytes,
)
from .decoder import generate
from .experiment_tracking import ExperimentTracker, JsonlFileMetricSink, LoggerMetricSink
from .training import CustomAdamW, TrainingConfig, load_checkpoint, train_language_model
from .transformer import TransformerLM

END_OF_TEXT_TOKEN = "<|endoftext|>"
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_D_MODEL = 512
DEFAULT_D_FF = 1344
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 16
DEFAULT_ROPE_THETA = 10_000.0
DEFAULT_TOTAL_TOKENS = 327_680_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_WARMUP_STEPS = 1_000
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_MIN_LEARNING_RATE = 3e-5
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_LOG_INTERVAL = 50
DEFAULT_EVAL_INTERVAL = 500
DEFAULT_EVAL_BATCHES = 20
DEFAULT_CHECKPOINT_INTERVAL = 1_000
DEFAULT_VAL_FRACTION = 0.01
DEFAULT_NUM_WORKERS = max(1, (os.cpu_count() or 1) - 1)


@dataclass(slots=True)
class TinyStoriesModelConfig:
    vocab_size: int = DEFAULT_VOCAB_SIZE
    context_length: int = DEFAULT_CONTEXT_LENGTH
    d_model: int = DEFAULT_D_MODEL
    d_ff: int = DEFAULT_D_FF
    num_layers: int = DEFAULT_NUM_LAYERS
    num_heads: int = DEFAULT_NUM_HEADS
    rope_theta: float = DEFAULT_ROPE_THETA


@dataclass(slots=True)
class TinyStoriesTrainConfig:
    data_dir: str
    output_dir: str
    batch_size: int = DEFAULT_BATCH_SIZE
    total_tokens: int = DEFAULT_TOTAL_TOKENS
    learning_rate: float = DEFAULT_LEARNING_RATE
    min_learning_rate: float = DEFAULT_MIN_LEARNING_RATE
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM
    log_interval: int = DEFAULT_LOG_INTERVAL
    eval_interval: int = DEFAULT_EVAL_INTERVAL
    eval_batches: int = DEFAULT_EVAL_BATCHES
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume_from: str | None = None


def _split_documents(text: str, eot_token: str = END_OF_TEXT_TOKEN) -> list[str]:
    documents = text.split(eot_token)
    return [document for document in documents if document]


def _chunked[T](items: list[T], chunk_size: int) -> list[list[T]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _count_pretokens_for_documents(documents: list[str]) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()
    for document in documents:
        for match in GPT2_PRETOKEN_PATTERN.finditer(document):
            piece = match.group(0)
            if piece:
                counts[_token_bytes(piece)] += 1
    return counts


def _build_bpe_from_token_frequencies(
    token_frequencies: Counter[tuple[bytes, ...]],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    unique_special_tokens = _dedupe_nonempty(special_tokens)
    min_required_vocab = 256 + len(unique_special_tokens)
    if vocab_size < min_required_vocab:
        raise ValueError(f"vocab_size must be at least {min_required_vocab}, got {vocab_size}.")

    vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
    next_token_id = 256
    for token in unique_special_tokens:
        vocab[next_token_id] = token.encode("utf-8")
        next_token_id += 1

    words: dict[int, _WordEntry] = {}
    inverted_index: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    pair_frequencies: Counter[tuple[bytes, bytes]] = Counter()

    for word_id, (token_bytes, frequency) in enumerate(token_frequencies.items()):
        symbols = list(token_bytes)
        words[word_id] = _WordEntry(symbols=symbols, frequency=frequency)
        for pair, pair_count in _count_pairs(symbols).items():
            pair_frequencies[pair] += pair_count * frequency
            inverted_index[pair].add(word_id)

    merges: list[tuple[bytes, bytes]] = []

    while next_token_id < vocab_size and pair_frequencies:
        best_pair: tuple[bytes, bytes] | None = None
        best_frequency = 1
        for pair, frequency in pair_frequencies.items():
            if frequency < best_frequency:
                continue
            if frequency > best_frequency or (best_pair is not None and pair > best_pair) or best_pair is None:
                best_pair = pair
                best_frequency = frequency

        if best_pair is None or best_frequency <= 1:
            break

        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[next_token_id] = merged_token
        next_token_id += 1

        affected_word_ids = list(inverted_index.get(best_pair, ()))
        for word_id in affected_word_ids:
            entry = words[word_id]
            old_pair_counts = _count_pairs(entry.symbols)
            for pair, pair_count in old_pair_counts.items():
                updated_frequency = pair_frequencies[pair] - pair_count * entry.frequency
                if updated_frequency > 0:
                    pair_frequencies[pair] = updated_frequency
                else:
                    pair_frequencies.pop(pair, None)
                word_ids = inverted_index.get(pair)
                if word_ids is not None:
                    word_ids.discard(word_id)
                    if not word_ids:
                        inverted_index.pop(pair, None)

            entry.symbols = _merge_symbols(entry.symbols, best_pair, merged_token)

            new_pair_counts = _count_pairs(entry.symbols)
            for pair, pair_count in new_pair_counts.items():
                pair_frequencies[pair] += pair_count * entry.frequency
                inverted_index[pair].add(word_id)

    return vocab, merges


def train_tinystories_bpe(
    corpus_path: str | os.PathLike,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    special_tokens: list[str] | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = _dedupe_nonempty(special_tokens or [END_OF_TEXT_TOKEN])
    with Path(corpus_path).open("r", encoding="utf-8") as handle:
        corpus = handle.read()

    documents = _split_documents(corpus, eot_token=END_OF_TEXT_TOKEN)
    if not documents:
        raise ValueError(f"No documents found in {corpus_path}.")

    worker_count = max(1, min(num_workers, len(documents)))
    shard_size = max(1, math.ceil(len(documents) / worker_count))
    shards = _chunked(documents, shard_size)

    if worker_count == 1:
        token_frequencies = _count_pretokens_for_documents(documents)
    else:
        with mp.Pool(processes=worker_count) as pool:
            partial_counts = pool.map(_count_pretokens_for_documents, shards)
        token_frequencies = Counter()
        for partial_count in partial_counts:
            token_frequencies.update(partial_count)

    return _build_bpe_from_token_frequencies(
        token_frequencies=token_frequencies,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )


def save_tokenizer_artifacts(
    *,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
    output_path: str | os.PathLike,
) -> None:
    payload = {
        "special_tokens": list(special_tokens),
        "vocab": {str(token_id): list(token_bytes) for token_id, token_bytes in vocab.items()},
        "merges": [[list(left), list(right)] for left, right in merges],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_tokenizer_artifacts(path: str | os.PathLike) -> Tokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = {int(token_id): bytes(token_bytes) for token_id, token_bytes in payload["vocab"].items()}
    merges = [(bytes(left), bytes(right)) for left, right in payload["merges"]]
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=payload["special_tokens"])


def build_tinystories_model(config: TinyStoriesModelConfig | None = None) -> TransformerLM:
    config = config or TinyStoriesModelConfig()
    return TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    )


def compute_total_steps(total_tokens: int, batch_size: int, context_length: int) -> int:
    tokens_per_step = batch_size * context_length
    if total_tokens % tokens_per_step != 0:
        raise ValueError(
            "total_tokens must be divisible by batch_size * context_length for an exact token budget. "
            f"Got total_tokens={total_tokens}, batch_size={batch_size}, context_length={context_length}."
        )
    return total_tokens // tokens_per_step


def _docs_to_token_ids(tokenizer: Tokenizer, documents: list[str], eot_token_id: int) -> np.ndarray:
    tokens: list[int] = []
    for document in documents:
        tokens.extend(tokenizer.encode(document))
        tokens.append(eot_token_id)
    if not tokens:
        raise ValueError("Document split produced an empty token sequence.")
    return np.asarray(tokens, dtype=np.uint16)


def preprocess_tinystories(
    *,
    corpus_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    dev_corpus_path: str | os.PathLike | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict[str, Any]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError(f"val_fraction must be in (0, 0.5), got {val_fraction}.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output_root / "tokenizer.json"
    metadata_path = output_root / "preprocess_metadata.json"
    train_bin_path = output_root / "train.bin"
    val_bin_path = output_root / "val.bin"

    vocab, merges = train_tinystories_bpe(
        corpus_path=corpus_path,
        vocab_size=vocab_size,
        special_tokens=[END_OF_TEXT_TOKEN],
        num_workers=num_workers,
    )
    save_tokenizer_artifacts(
        vocab=vocab,
        merges=merges,
        special_tokens=[END_OF_TEXT_TOKEN],
        output_path=tokenizer_path,
    )
    tokenizer = load_tokenizer_artifacts(tokenizer_path)
    eot_token_id = tokenizer.encode(END_OF_TEXT_TOKEN)[0]

    with Path(corpus_path).open("r", encoding="utf-8") as handle:
        train_documents = _split_documents(handle.read(), eot_token=END_OF_TEXT_TOKEN)

    if dev_corpus_path is not None:
        with Path(dev_corpus_path).open("r", encoding="utf-8") as handle:
            val_documents = _split_documents(handle.read(), eot_token=END_OF_TEXT_TOKEN)
    else:
        num_val_docs = max(1, int(len(train_documents) * val_fraction))
        num_train_docs = len(train_documents) - num_val_docs
        if num_train_docs <= 0:
            raise ValueError("Validation split consumed all documents; decrease val_fraction.")
        val_documents = train_documents[num_train_docs:]
        train_documents = train_documents[:num_train_docs]

    train_ids = _docs_to_token_ids(tokenizer, train_documents, eot_token_id=eot_token_id)
    val_ids = _docs_to_token_ids(tokenizer, val_documents, eot_token_id=eot_token_id)
    train_ids.tofile(train_bin_path)
    val_ids.tofile(val_bin_path)

    metadata = {
        "corpus_path": str(Path(corpus_path).resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "train_bin_path": str(train_bin_path.resolve()),
        "val_bin_path": str(val_bin_path.resolve()),
        "dev_corpus_path": str(Path(dev_corpus_path).resolve()) if dev_corpus_path is not None else None,
        "vocab_size": vocab_size,
        "num_merges": len(merges),
        "num_documents": len(train_documents) + len(val_documents),
        "num_train_documents": len(train_documents),
        "num_val_documents": len(val_documents),
        "train_tokens": int(train_ids.shape[0]),
        "val_tokens": int(val_ids.shape[0]),
        "special_tokens": [END_OF_TEXT_TOKEN],
        "dtype": "uint16",
        "num_workers": num_workers,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def train_tinystories_model(
    train_config: TinyStoriesTrainConfig,
    model_config: TinyStoriesModelConfig | None = None,
) -> dict[str, Any]:
    model_config = model_config or TinyStoriesModelConfig()
    output_root = Path(train_config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    max_steps = compute_total_steps(
        total_tokens=train_config.total_tokens,
        batch_size=train_config.batch_size,
        context_length=model_config.context_length,
    )
    metrics_path = output_root / "metrics.jsonl"
    checkpoint_path = output_root / "checkpoint.pt"
    config_path = output_root / "train_config.json"

    tracker = ExperimentTracker(
        sinks=[
            LoggerMetricSink(),
            JsonlFileMetricSink(metrics_path),
        ]
    )
    training_config = TrainingConfig(
        train_data_path=Path(train_config.data_dir) / "train.bin",
        val_data_path=Path(train_config.data_dir) / "val.bin",
        batch_size=train_config.batch_size,
        context_length=model_config.context_length,
        learning_rate=train_config.learning_rate,
        min_learning_rate=train_config.min_learning_rate,
        warmup_steps=train_config.warmup_steps,
        max_steps=max_steps,
        weight_decay=train_config.weight_decay,
        grad_clip_norm=train_config.grad_clip_norm,
        log_interval=train_config.log_interval,
        eval_interval=train_config.eval_interval,
        eval_batches=train_config.eval_batches,
        checkpoint_interval=train_config.checkpoint_interval,
        checkpoint_path=checkpoint_path,
        resume_from=train_config.resume_from,
        device=train_config.device,
        experiment_tracker=tracker,
    )

    payload = {
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "resolved_max_steps": max_steps,
    }
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    model = build_tinystories_model(model_config)
    final_step = train_language_model(model=model, config=training_config)
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "config_path": str(config_path.resolve()),
        "final_step": final_step,
    }


def load_model_from_checkpoint(
    *,
    checkpoint_path: str | os.PathLike,
    model_config: TinyStoriesModelConfig | None = None,
) -> TransformerLM:
    model_config = model_config or TinyStoriesModelConfig()
    model = build_tinystories_model(model_config)
    optimizer = CustomAdamW(model.parameters(), lr=1.0)
    load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
    return model


def _qualitative_commentary(*, temperature: float, top_p: float, generated_tokens: int) -> str:
    style_bits: list[str] = []
    if temperature < 0.8:
        style_bits.append("较低温度通常会让文本更稳定、更保守，重复风险也更低。")
    elif temperature > 1.1:
        style_bits.append("较高温度会提高新颖性，但也更容易带来局部跑题或语法抖动。")
    else:
        style_bits.append("中等温度通常能在稳定性和多样性之间取得较平衡的表现。")

    if top_p < 0.9:
        style_bits.append("更小的 top-p 会明显收紧候选集，减少低概率噪声。")
    elif top_p >= 0.98:
        style_bits.append("较大的 top-p 会保留更多长尾词，风格更自由，但也更容易发散。")
    else:
        style_bits.append("当前 top-p 设置属于常见折中区间，通常能兼顾流畅度与变化性。")

    style_bits.append(f"本次导出共生成 {generated_tokens} 个新 token，可用于提交实验中的定性分析部分。")
    return " ".join(style_bits)


def generate_tinystories_text(
    *,
    checkpoint_path: str | os.PathLike,
    tokenizer_path: str | os.PathLike,
    prompt: str,
    output_path: str | os.PathLike,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    top_p: float = 0.95,
    model_config: TinyStoriesModelConfig | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    tokenizer = load_tokenizer_artifacts(tokenizer_path)
    model = load_model_from_checkpoint(checkpoint_path=checkpoint_path, model_config=model_config)
    model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    eos_token_id = tokenizer.encode(END_OF_TEXT_TOKEN)[0]
    output_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
    )[0].tolist()

    generated_ids = output_ids[len(prompt_ids) :]
    text = tokenizer.decode(output_ids)
    commentary = _qualitative_commentary(
        temperature=temperature,
        top_p=top_p,
        generated_tokens=len(generated_ids),
    )

    output_root = Path(output_path)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.write_text(text, encoding="utf-8")
    summary_path = output_root.with_suffix(".json")
    summary_payload = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "tokenizer_path": str(Path(tokenizer_path).resolve()),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "generated_token_count": len(generated_ids),
        "stopped_on_eos": bool(generated_ids and generated_ids[-1] == eos_token_id),
        "qualitative_commentary": commentary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "text_path": str(output_root.resolve()),
        "summary_path": str(summary_path.resolve()),
        "generated_token_count": len(generated_ids),
        "qualitative_commentary": commentary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TinyStories preprocessing, training, and generation helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Train BPE and serialize TinyStories tokens.")
    preprocess_parser.add_argument("--corpus-path", required=True)
    preprocess_parser.add_argument("--output-dir", required=True)
    preprocess_parser.add_argument("--dev-corpus-path")
    preprocess_parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    preprocess_parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    preprocess_parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)

    train_parser = subparsers.add_parser("train", help="Train the TinyStories language model.")
    train_parser.add_argument("--data-dir", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train_parser.add_argument("--total-tokens", type=int, default=DEFAULT_TOTAL_TOKENS)
    train_parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    train_parser.add_argument("--min-learning-rate", type=float, default=DEFAULT_MIN_LEARNING_RATE)
    train_parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    train_parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    train_parser.add_argument("--grad-clip-norm", type=float, default=DEFAULT_GRAD_CLIP_NORM)
    train_parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    train_parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    train_parser.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    train_parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--resume-from")

    generate_parser = subparsers.add_parser("generate", help="Generate text from a trained TinyStories checkpoint.")
    generate_parser.add_argument("--checkpoint-path", required=True)
    generate_parser.add_argument("--tokenizer-path", required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--output-path", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=256)
    generate_parser.add_argument("--temperature", type=float, default=0.9)
    generate_parser.add_argument("--top-p", type=float, default=0.95)
    generate_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        result = preprocess_tinystories(
            corpus_path=args.corpus_path,
            output_dir=args.output_dir,
            dev_corpus_path=args.dev_corpus_path,
            vocab_size=args.vocab_size,
            val_fraction=args.val_fraction,
            num_workers=args.num_workers,
        )
    elif args.command == "train":
        result = train_tinystories_model(
            TinyStoriesTrainConfig(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                total_tokens=args.total_tokens,
                learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                warmup_steps=args.warmup_steps,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                log_interval=args.log_interval,
                eval_interval=args.eval_interval,
                eval_batches=args.eval_batches,
                checkpoint_interval=args.checkpoint_interval,
                device=args.device,
                resume_from=args.resume_from,
            )
        )
    else:
        result = generate_tinystories_text(
            checkpoint_path=args.checkpoint_path,
            tokenizer_path=args.tokenizer_path,
            prompt=args.prompt,
            output_path=args.output_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
