# Centralized configuration for model, training, and inference
# CPU-OPTIMIZED VERSION

# Model Presets
TINY_MODEL = {
    'vocab_size': 50258,
    'hidden_size': 128,
    'num_layers': 2,
    'num_heads': 4,
    'ff_expansion': 2,
    'num_experts': 2,
    'moe_k': 1,
    'seq_len': 256,
}

SMALL_MODEL = {
    'vocab_size': 50258,
    'hidden_size': 256,
    'num_layers': 4,
    'num_heads': 8,
    'ff_expansion': 2,
    'num_experts': 4,
    'moe_k': 2,
    'seq_len': 512,
}

MEDIUM_MODEL = {
    'vocab_size': 50258,
    'hidden_size': 512,
    'num_layers': 6,
    'num_heads': 8,
    'ff_expansion': 4,
    'num_experts': 8,
    'moe_k': 2,
    'seq_len': 1024,
}

# Default to TINY for CPU (won't run out of memory)
MODEL_CONFIG = TINY_MODEL

TRAIN_CONFIG = {
    'batch_size': 1,  # CRITICAL: Use batch_size=1 on CPU to save memory
    'lr': 1e-3,  # Higher LR for CPU training
    'epochs': 2,  # Reduced for testing
    'clip': 0.5,  # Lower clipping threshold
    'ema_decay': 0.99,  # Faster EMA decay
    'val_split': 0.2,  # More validation data
    'checkpoint_path': 'utils/checkpoint.pt',
    'warmup_steps': 10,  # Fewer warmup steps
    'patience': 2,  # Early stopping after 2 epochs
    'gradient_accumulation_steps': 1,  # No accumulation needed
    'log_interval': 1,  # Log every batch
}

INFER_CONFIG = {
    'max_response_length': 50,  # Shorter generations on CPU
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 20,  # Smaller top-k for faster inference
}