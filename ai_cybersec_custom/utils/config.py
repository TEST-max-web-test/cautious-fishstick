# Centralized configuration for model, training, and inference
# Includes preset sizes: small, medium, large

# Model Presets
SMALL_MODEL = {
    'vocab_size': 50258,
    'hidden_size': 256,
    'num_layers': 4,
    'num_heads': 8,
    'ff_expansion': 2,
    'num_experts': 4,
    'moe_k': 2,
    'seq_len': 1024,
}

MEDIUM_MODEL = {
    'vocab_size': 50258,
    'hidden_size': 512,
    'num_layers': 6,
    'num_heads': 8,
    'ff_expansion': 4,
    'num_experts': 8,
    'moe_k': 2,
    'seq_len': 2048,
}

LARGE_MODEL = {
    'vocab_size': 50258,
    'hidden_size': 1024,
    'num_layers': 12,
    'num_heads': 16,
    'ff_expansion': 4,
    'num_experts': 16,
    'moe_k': 2,
    'seq_len': 4096,
}

# Default to SMALL for fast testing
MODEL_CONFIG = SMALL_MODEL

TRAIN_CONFIG = {
    'batch_size': 4,
    'lr': 3e-4,
    'epochs': 5,  # Increased from 1 for better training
    'clip': 1.0,
    'ema_decay': 0.999,
    'val_split': 0.1,
    'checkpoint_path': 'utils/checkpoint.pt',
    'warmup_steps': 500,  # NEW: Learning rate warmup
    'patience': 3,  # NEW: Early stopping patience
    'gradient_accumulation_steps': 1,  # NEW: For larger effective batch size
    'log_interval': 10,  # NEW: Logging interval
}

INFER_CONFIG = {
    'max_response_length': 100,  # Increased from 50
    'temperature': 0.8,  # Lowered from 1.0 for more focused responses
    'top_p': 0.9,  # Lowered from 0.95
    'top_k': 40,  # NEW: Top-k sampling
}