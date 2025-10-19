# ai_cybersec_custom/utils/config.py
# PRODUCTION CONFIGURATION - ENTERPRISE GRADE

MODEL_CONFIG = {
    # Model architecture
    'vocab_size': 2000,
    'hidden_size': 96,
    'num_layers': 3,
    'num_heads': 3,
    'ff_expansion': 4,
    'dropout': 0.3,
    'seq_len': 512,
}

SPECIAL_TOKENS = {
    'PAD': 0,
    'UNK': 1,
    'BOS': 2,
    'EOS': 3,
}

TRAIN_CONFIG = {
    # Training hyperparameters
    'batch_size': 4,
    'lr': 3e-4,
    'epochs': 50,
    'clip': 1.0,
    'warmup_steps': 100,
    'patience': 10,
    'weight_decay': 0.01,
    'label_smoothing': 0.1,
    'checkpoint_path': 'ai_cybersec_custom/train/HERE/checkpoint.pt',
}

INFER_CONFIG = {
    # Generation hyperparameters
    'max_response_length': 200,
    'temperature': 0.3,
    'top_p': 0.9,
    'top_k': 50,
    'repetition_penalty': 1.1,
}