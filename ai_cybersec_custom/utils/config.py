# PRODUCTION CONFIGURATION
# Optimized hyperparameters based on modern LLM research

MODEL_CONFIG = {
    # Model architecture
    'vocab_size': 2000,
    'hidden_size': 256,      # Increased from 128 for better capacity
    'num_layers': 6,         # Increased from 4 for more depth
    'num_heads': 8,          # Increased from 4 for better attention
    'ff_expansion': 4,       # Standard expansion factor
    'dropout': 0.1,          # Moderate dropout
    'seq_len': 512,          # Maximum sequence length
}

SPECIAL_TOKENS = {
    'PAD': 0,
    'UNK': 1,
    'BOS': 2,
    'EOS': 3,
}

TRAIN_CONFIG = {
    # Training hyperparameters
    'batch_size': 8,          # Increased from 4 for GPU efficiency
    'lr': 3e-4,               # Standard learning rate for transformers
    'epochs': 100,            # More epochs with early stopping
    'clip': 1.0,              # Gradient clipping
    'warmup_steps': 100,      # Linear warmup
    'patience': 20,           # Early stopping patience
    'weight_decay': 0.01,     # L2 regularization
    'label_smoothing': 0.1,   # Prevent overconfidence
    'log_interval': 10,       # Log every N batches
    
    # Paths
    'checkpoint_path': 'ai_cybersec_custom/train/HERE/checkpoint.pt',
}

INFER_CONFIG = {
    # Generation hyperparameters
    'max_response_length': 150,  # Increased for longer responses
    'temperature': 0.8,          # Lower = more focused, Higher = more creative
    'top_p': 0.9,                # Nucleus sampling threshold
    'top_k': 50,                 # Top-k sampling
    'repetition_penalty': 1.15,  # Reduced from 1.2 - was too aggressive
}