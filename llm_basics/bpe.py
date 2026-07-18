from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import regex

# GPT-2 风格的预分词正则。
# 这里要保留很多 token 类别前面的空格，否则训练出的 merge 顺序会和 GPT-2 不一致。
GPT2_PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _dedupe_nonempty(tokens: list[str] | None) -> list[str]:
    """按首次出现顺序去重，并丢弃空字符串。"""

    seen: set[str] = set()
    result: list[str] = []
    for token in tokens or []:
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def _compile_special_pattern(special_tokens: list[str]) -> regex.Pattern[str] | None:
    """构造 special token 的正则，并优先匹配更长的 token。"""

    if not special_tokens:
        return None
    escaped = [regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    return regex.compile("|".join(escaped))


def _iter_segments(text: str, special_pattern: regex.Pattern[str] | None) -> Iterator[tuple[bool, str]]:
    """把文本切成普通片段和 special token 片段。"""

    if not text:
        return
    if special_pattern is None:
        yield False, text
        return

    last_end = 0
    for match in special_pattern.finditer(text):
        start, end = match.span()
        if start > last_end:
            yield False, text[last_end:start]
        yield True, match.group(0)
        last_end = end
    if last_end < len(text):
        yield False, text[last_end:]


def _iter_pretokens(text: str, special_pattern: regex.Pattern[str] | None) -> Iterator[tuple[bool, str]]:
    """从文本中依次产出 special token 或 GPT-2 预分词结果。"""

    for is_special, segment in _iter_segments(text, special_pattern):
        if is_special:
            yield True, segment
            continue
        for match in GPT2_PRETOKEN_PATTERN.finditer(segment):
            yield False, match.group(0)


def _token_bytes(text: str) -> tuple[bytes, ...]:
    """把字符串编码成 UTF-8，并拆成单字节符号序列。"""

    data = text.encode("utf-8")
    return tuple(bytes([value]) for value in data)


def _count_pairs(symbols: list[bytes]) -> Counter[tuple[bytes, bytes]]:
    """统计一个符号序列内部所有相邻 pair 的出现次数。"""

    return Counter(zip(symbols, symbols[1:]))


def _merge_symbols(symbols: list[bytes], pair: tuple[bytes, bytes], merged: bytes) -> list[bytes]:
    """把序列里所有不重叠的目标 pair 合并成一个新符号。"""

    if len(symbols) < 2:
        return symbols

    left, right = pair
    merged_symbols: list[bytes] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and symbols[index] == left and symbols[index + 1] == right:
            merged_symbols.append(merged)
            index += 2
        else:
            merged_symbols.append(symbols[index])
            index += 1
    return merged_symbols


@dataclass(slots=True)
class _WordEntry:
    """训练阶段对一个唯一 pre-token 的可变表示。"""

    symbols: list[bytes]
    frequency: int


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从 UTF-8 文本语料中训练 byte-level BPE 的词表和 merge 列表。

    训练流程如下：
    1. 用 256 个单字节 token 加 special token 初始化词表。
    2. 先按 special token 边界切分，再用 GPT-2 正则做预分词。
    3. 把每个 pre-token 转成字节序列并统计频次。
    4. 反复合并最高频的相邻字节对，直到词表满了，或不存在频次大于 1 的 pair。
    """

    unique_special_tokens = _dedupe_nonempty(special_tokens)
    min_required_vocab = 256 + len(unique_special_tokens)
    if vocab_size < min_required_vocab:
        raise ValueError(f"vocab_size must be at least {min_required_vocab}, got {vocab_size}.")

    # 基础词表固定是 0~255 的全部单字节。
    vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
    next_token_id = 256
    for token in unique_special_tokens:
        vocab[next_token_id] = token.encode("utf-8")
        next_token_id += 1

    special_pattern = _compile_special_pattern(unique_special_tokens)
    token_frequencies: Counter[tuple[bytes, ...]] = Counter()

    with open(input_path, encoding="utf-8") as handle:
        corpus = handle.read()

    for is_special, piece in _iter_pretokens(corpus, special_pattern):
        if is_special:
            continue
        if piece:
            token_frequencies[_token_bytes(piece)] += 1

    # `words` 只保存每个唯一 pre-token 一份。
    # `pair_frequencies` 保存全局 pair 频次。
    # `inverted_index` 记录某个 pair 出现在哪些词里，便于增量更新。
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
            # 平局时选择字节字典序更大的 pair。
            if frequency > best_frequency or (best_pair is not None and pair > best_pair) or best_pair is None:
                best_pair = pair
                best_frequency = frequency

        if best_pair is None or best_frequency <= 1:
            break

        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[next_token_id] = merged_token
        next_token_id += 1

        # 只有包含当前 best_pair 的词会受影响，因此只增量更新这些词，
        # 不需要每一轮都重新扫描整份语料。
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


class Tokenizer:
    """带 GPT-2 风格预分词的 byte-level BPE 分词器。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        # 运行期把字节片段映射回 token id 时使用的反向索引。
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        # rank 越小，说明这个 merge 学得越早，优先级越高。
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_tokens = _dedupe_nonempty(special_tokens)
        self.special_pattern = _compile_special_pattern(self.special_tokens)

    def _apply_bpe(self, token_bytes: bytes) -> list[bytes]:
        """对一个 pre-token 的字节序列贪心应用已学习的 merges。"""

        symbols = [bytes([value]) for value in token_bytes]
        while len(symbols) > 1:
            best_index = -1
            best_rank: int | None = None
            for index in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[index], symbols[index + 1]))
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = index
            if best_index < 0:
                break
            merged = symbols[best_index] + symbols[best_index + 1]
            symbols[best_index : best_index + 2] = [merged]
        return symbols

    def encode(self, text: str) -> list[int]:
        """把完整字符串编码成 token id 序列。"""

        if not text:
            return []

        token_ids: list[int] = []
        for is_special, piece in _iter_pretokens(text, self.special_pattern):
            if is_special:
                token_ids.append(self.bytes_to_id[piece.encode("utf-8")])
                continue
            # 普通文本会先做 GPT-2 预分词，再对每个 pre-token 做贪心 BPE 合并，
            # 最后映射成 token id。
            for symbol in self._apply_bpe(piece.encode("utf-8")):
                token_ids.append(self.bytes_to_id[symbol])
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """按块懒惰编码文本可迭代对象，内部复用 ``encode``。"""

        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """把 token id 还原成字节串，再一次性解码成 UTF-8 字符串。"""

        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
