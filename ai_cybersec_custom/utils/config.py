# MINIMAL WORKING CONFIGURATION
# Copy this to: ai_cybersec_custom/utils/config.py

MODEL_CONFIG = {
    'vocab_size': 2000,
    'hidden_size': 128,
    'num_layers': 4,
    'num_heads': 4,
    'ff_expansion': 4,
    'dropout': 0.1,
    'seq_len': 512,
}

SPECIAL_TOKENS = {
    'PAD': 0,
    'UNK': 1,
    'BOS': 2,
    'EOS': 3,
}

TRAIN_CONFIG = {
    'batch_size': 4,
    'lr': 3e-4,
    'epochs': 50,  # Reduced for testing
    'clip': 1.0,
    'checkpoint_path': 'ai_cybersec_custom/train/HERE/checkpoint.pt',
    'warmup_steps': 50,
    'patience': 15,
    'gradient_accumulation_steps': 1,
    'log_interval': 5,
    'weight_decay': 0.01,
}

INFER_CONFIG = {
    'max_response_length': 100,
    'temperature': 0.8,
    'top_p': 0.92,
    'top_k': 50,
    'repetition_penalty': 1.2,
}