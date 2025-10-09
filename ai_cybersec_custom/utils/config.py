# Centralized configuration for model, training, and inference

MODEL_CONFIG = {
    'vocab_size': 50258,
    'hidden_size': 64,
    'num_layers': 2,
    'num_heads': 2,
    'ff_expansion': 2,
    'num_experts': 4,
    'moe_k': 2,
    'seq_len': 2048
}

TRAIN_CONFIG = {
    'batch_size': 4,
    'lr': 3e-4,
    'epochs': 1,
    'clip': 1.0,
    'ema_decay': 0.999,
    'val_split': 0.1,
    'checkpoint_path': 'utils/checkpoint.pt'
}

INFER_CONFIG = {
    'max_response_length': 50,
    'temperature': 1.0,
    'top_p': 0.95
}
