# TinyStories Server Runbook

This repository already contains the TinyStories experiment code you need. The pipeline is:

1. preprocess text into `tokenizer.json`, `train.bin`, and `val.bin`
2. train the language model
3. generate a sample text from the trained checkpoint

## Manual commands

### 0. Install dependencies

```bash
cd /assignment1
uv sync
```

### 1. Preprocess data

If you already have plain-text train and validation files separated by `<|endoftext|>`, run:

```bash
python -m llm_basics.tinystories preprocess \
  --corpus-path /root/autodl-tmp/TinyStories/TinyStories-train.txt \
  --dev-corpus-path /root/autodl-tmp/TinyStories/TinyStories-valid.txt \
  --output-dir /root/autodl-tmp/tinystories_data \
  --vocab-size 10000 \
  --num-workers 8
```

Outputs:

- `/root/autodl-tmp/tinystories_data/tokenizer.json`
- `/root/autodl-tmp/tinystories_data/train.bin`
- `/root/autodl-tmp/tinystories_data/val.bin`
- `/root/autodl-tmp/tinystories_data/preprocess_metadata.json`

### 2. Train model

```bash
python -m llm_basics.tinystories train \
  --data-dir /root/autodl-tmp/tinystories_data \
  --output-dir /root/autodl-tmp/tinystories_run \
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

Outputs:

- `/root/autodl-tmp/tinystories_run/checkpoint.pt`
- `/root/autodl-tmp/tinystories_run/metrics.jsonl`
- `/root/autodl-tmp/tinystories_run/train_config.json`

### 3. Resume training

```bash
python -m llm_basics.tinystories train \
  --data-dir /root/autodl-tmp/tinystories_data \
  --output-dir /root/autodl-tmp/tinystories_run \
  --batch-size 32 \
  --total-tokens 327680000 \
  --device cuda \
  --resume-from /root/autodl-tmp/tinystories_run/checkpoint.pt
```

### 4. Generate text

```bash
python -m llm_basics.tinystories generate \
  --checkpoint-path /root/autodl-tmp/tinystories_run/checkpoint.pt \
  --tokenizer-path /root/autodl-tmp/tinystories_data/tokenizer.json \
  --prompt "Once upon a time, a little girl found a secret in the forest." \
  --output-path /root/autodl-tmp/tinystories_run/sample.txt \
  --max-new-tokens 256 \
  --temperature 0.9 \
  --top-p 0.95 \
  --device cuda
```

Outputs:

- `/root/autodl-tmp/tinystories_run/sample.txt`
- `/root/autodl-tmp/tinystories_run/sample.json`

## AutoDL one-command script

For AutoDL, the repository also includes:

- `/assignment1/run_tinystories_autodl.sh`

This script runs the full pipeline in order:

1. preprocess
2. train
3. generate
4. shut down the server on success

Run it with:

```bash
cd /assignment1
bash run_tinystories_autodl.sh
```

Disable auto shutdown for one run:

```bash
cd /assignment1
AUTO_SHUTDOWN_ON_SUCCESS=0 bash run_tinystories_autodl.sh
```

Override the prompt:

```bash
cd /assignment1
PROMPT="Once upon a time, a small fox found a lantern in the snow." bash run_tinystories_autodl.sh
```

## Reused modules

The experiment pipeline reuses your existing `llm_basics` components:

- `llm_basics.bpe.Tokenizer`
- `llm_basics.transformer.TransformerLM`
- `llm_basics.training.CustomAdamW`
- `llm_basics.training.train_language_model`
- `llm_basics.training.load_checkpoint`
- `llm_basics.training.save_checkpoint`
- `llm_basics.decoder.generate`

The new `llm_basics.tinystories` module mainly acts as the experiment entrypoint around those components.
