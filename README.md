# LLM Basics
Implementation of basic LLM modules.

## TinyStories Experiment Pipeline

This repository now includes a local-first TinyStories experiment pipeline that covers:

- byte-level BPE training with `<|endoftext|>` as a hard document boundary
- multiprocessing pre-token counting for faster tokenizer training
- `uint16` serialization for `train.bin` and `val.bin`
- `np.memmap`-backed training data loading
- decoder-only Transformer training with RoPE, RMSNorm, and SwiGLU
- checkpoint save / resume
- text generation with temperature and top-p sampling

### 1. Preprocess TinyStories

```bash
python -m llm_basics.tinystories preprocess \
  --corpus-path /path/to/TinyStoriesV2-GPT4-train.txt \
  --dev-corpus-path /path/to/TinyStoriesV2-GPT4-valid.txt \
  --output-dir /path/to/tinystories_data \
  --vocab-size 10000 \
  --num-workers 8
```

This writes:

- `tokenizer.json`
- `train.bin`
- `val.bin`
- `preprocess_metadata.json`

If you do not have a separate development file, omit `--dev-corpus-path` and the script will
fall back to splitting the training corpus with `--val-fraction`.

### 2. Train the Language Model

```bash
python -m llm_basics.tinystories train \
  --data-dir /path/to/tinystories_data \
  --output-dir /path/to/tinystories_runs/base \
  --batch-size 32 \
  --total-tokens 327680000 \
  --device cuda
```

Default model configuration matches the requested setup:

- `vocab_size=10000`
- `context_length=256`
- `d_model=512`
- `d_ff=1344`
- `num_layers=4`
- `num_heads=16`
- `rope_theta=10000`

Training writes:

- `checkpoint.pt`
- `metrics.jsonl`
- `train_config.json`

### 3. Resume Training

```bash
python -m llm_basics.tinystories train \
  --data-dir /path/to/tinystories_data \
  --output-dir /path/to/tinystories_runs/base \
  --batch-size 32 \
  --total-tokens 327680000 \
  --device cuda \
  --resume-from /path/to/tinystories_runs/base/checkpoint.pt
```

### 4. Generate Text

```bash
python -m llm_basics.tinystories generate \
  --checkpoint-path /path/to/tinystories_runs/base/checkpoint.pt \
  --tokenizer-path /path/to/tinystories_data/tokenizer.json \
  --prompt "Once upon a time" \
  --output-path /path/to/tinystories_runs/base/sample.txt \
  --max-new-tokens 256 \
  --temperature 0.9 \
  --top-p 0.95 \
  --device cuda
```

This writes:

- generated text to `sample.txt`
- a short qualitative summary to `sample.json`
