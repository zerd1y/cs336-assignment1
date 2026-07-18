# TinyStories Server Runbook

这份清单假设你已经把当前仓库完整上传到服务器，并且 TinyStories 原始文本文件也已经放好。

## 0. 环境准备

```bash
cd /path/to/assignment1
uv sync
```

如果服务器不用 `uv`，也可以直接用现有虚拟环境等价安装依赖。

## 1. 预处理数据

原始语料假设为单文件，并且文档之间由 `<|endoftext|>` 分隔。

```bash
python -m llm_basics.tinystories preprocess \
  --corpus-path /path/to/TinyStoriesV2-GPT4-train.txt \
  --dev-corpus-path /path/to/TinyStoriesV2-GPT4-valid.txt \
  --output-dir /path/to/outputs/tinystories_data \
  --vocab-size 10000 \
  --num-workers 8
```

预处理产物：

- `/path/to/outputs/tinystories_data/tokenizer.json`
- `/path/to/outputs/tinystories_data/train.bin`
- `/path/to/outputs/tinystories_data/val.bin`
- `/path/to/outputs/tinystories_data/preprocess_metadata.json`

如果你只有训练集单文件，没有单独的开发集文件，可以去掉 `--dev-corpus-path`，
此时脚本会按 `--val-fraction` 从训练集内部切出验证集。

## 2. 训练模型

下面这条命令使用你要求的默认结构：

- `vocab_size=10000`
- `context_length=256`
- `d_model=512`
- `d_ff=1344`
- `num_layers=4`
- `num_heads=16`
- `rope_theta=10000`

同时总 token 数默认为 `327680000`，对应：

- `batch_size=32`
- `context_length=256`
- `total_steps=40000`

```bash
python -m llm_basics.tinystories train \
  --data-dir /path/to/outputs/tinystories_data \
  --output-dir /path/to/outputs/tinystories_run \
  --batch-size 32 \
  --total-tokens 327680000 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-steps 1000 \
  --weight-decay 0.1 \
  --grad-clip-norm 1.0 \
  --log-interval 50 \
  --eval-interval 500 \
  --eval-batches 20 \
  --checkpoint-interval 1000 \
  --device cuda
```

训练产物：

- `/path/to/outputs/tinystories_run/checkpoint.pt`
- `/path/to/outputs/tinystories_run/metrics.jsonl`
- `/path/to/outputs/tinystories_run/train_config.json`

## 3. 断点续训

```bash
python -m llm_basics.tinystories train \
  --data-dir /path/to/outputs/tinystories_data \
  --output-dir /path/to/outputs/tinystories_run \
  --batch-size 32 \
  --total-tokens 327680000 \
  --device cuda \
  --resume-from /path/to/outputs/tinystories_run/checkpoint.pt
```

## 4. 生成至少 256 个 token

```bash
python -m llm_basics.tinystories generate \
  --checkpoint-path /path/to/outputs/tinystories_run/checkpoint.pt \
  --tokenizer-path /path/to/outputs/tinystories_data/tokenizer.json \
  --prompt "Once upon a time, a little rabbit found a blue door in the forest." \
  --output-path /path/to/outputs/tinystories_run/sample_t09_p095.txt \
  --max-new-tokens 256 \
  --temperature 0.9 \
  --top-p 0.95 \
  --device cuda
```

生成产物：

- `/path/to/outputs/tinystories_run/sample_t09_p095.txt`
- `/path/to/outputs/tinystories_run/sample_t09_p095.json`

其中 `.json` 会带一段可直接拿来写实验报告的定性点评草稿。

## 5. 建议至少补两组对照生成

更稳：

```bash
python -m llm_basics.tinystories generate \
  --checkpoint-path /path/to/outputs/tinystories_run/checkpoint.pt \
  --tokenizer-path /path/to/outputs/tinystories_data/tokenizer.json \
  --prompt "Once upon a time, a little rabbit found a blue door in the forest." \
  --output-path /path/to/outputs/tinystories_run/sample_t07_p09.txt \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 0.90 \
  --device cuda
```

更自由：

```bash
python -m llm_basics.tinystories generate \
  --checkpoint-path /path/to/outputs/tinystories_run/checkpoint.pt \
  --tokenizer-path /path/to/outputs/tinystories_data/tokenizer.json \
  --prompt "Once upon a time, a little rabbit found a blue door in the forest." \
  --output-path /path/to/outputs/tinystories_run/sample_t11_p098.txt \
  --max-new-tokens 256 \
  --temperature 1.1 \
  --top-p 0.98 \
  --device cuda
```

## 6. 你最终需要带走的文件

- 代码：整个当前仓库
- 训练数据产物：`tinystories_data/`
- 训练日志与权重：`tinystories_run/`
- 至少一份 256 token 以上生成文本

## 7. 这套流程复用了哪些已有模块

为了尽量复用你原先在 `llm_basics` 里的实现，这条实验链路直接使用了：

- `llm_basics.bpe.Tokenizer`
- `llm_basics.transformer.TransformerLM`
- `llm_basics.training.CustomAdamW`
- `llm_basics.training.train_language_model`
- `llm_basics.training.load_checkpoint`
- `llm_basics.training.save_checkpoint`
- `llm_basics.decoder.generate`

新增的 `llm_basics.tinystories` 主要只是把这些现有模块串成可直接跑实验的入口。
