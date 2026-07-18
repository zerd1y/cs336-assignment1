import importlib.metadata

from .bpe import Tokenizer, run_train_bpe
from .decoder import generate, sample_top_p
from .experiment_tracking import (
    CompositeMetricSink,
    ExperimentTracker,
    InMemoryMetricSink,
    JsonlFileMetricSink,
    LoggerMetricSink,
    MetricRecord,
    SummaryWriterMetricSink,
    WandbLikeMetricSink,
    build_default_experiment_tracker,
)
from .tinystories import (
    END_OF_TEXT_TOKEN,
    TinyStoriesModelConfig,
    TinyStoriesTrainConfig,
    build_tinystories_model,
    compute_total_steps,
    generate_tinystories_text,
    load_model_from_checkpoint,
    load_tokenizer_artifacts,
    preprocess_tinystories,
    save_tokenizer_artifacts,
    train_tinystories_bpe,
    train_tinystories_model,
)
from .training import (
    CustomAdamW,
    TrainingConfig,
    clip_gradients,
    compute_cross_entropy_loss,
    evaluate_language_model,
    get_batch,
    get_cosine_lr_with_warmup,
    load_checkpoint,
    load_token_memmap,
    save_checkpoint,
    train_language_model,
)
from .transformer import (
    CausalMultiHeadSelfAttention,
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    build_transformer_lm_from_state_dict,
    run_embedding,
    run_linear,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rmsnorm,
    run_rope,
    run_swiglu,
    run_transformer_block,
    run_transformer_lm,
    scaled_dot_product_attention,
    silu,
    softmax,
)

try:
    __version__ = importlib.metadata.version("llm_basics")
except importlib.metadata.PackageNotFoundError:
    pass
