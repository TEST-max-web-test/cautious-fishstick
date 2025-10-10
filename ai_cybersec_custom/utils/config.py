# Centralized configuration for model, training, and inference
# FIXED VERSION with proper special token IDs and settings

# Model Presets
TINY_MODEL = {
    'vocab_size': 8000,
    'hidden_size': 128,
    'num_layers': 2,
    'num_heads': 4,
    'ff_expansion': 2,
    'num_experts': 2,
    'moe_k': 1,
    'seq_len': 512,
}

SMALL_MODEL = {
    'vocab_size': 8000,
    'hidden_size': 256,
    'num_layers': 4,
    'num_heads': 8,
    'ff_expansion': 2,
    'num_experts': 4,
    'moe_k': 2,
    'seq_len': 512,
}

MEDIUM_MODEL = {
    'vocab_size': 8000,
    'hidden_size': 512,
    'num_layers': 6,
    'num_heads': 8,
    'ff_expansion': 4,
    'num_experts': 8,
    'moe_k': 2,
    'seq_len': 1024,
}

# Default to TINY for fast training
MODEL_CONFIG = TINY_MODEL

# ✅ FIXED: Special token IDs must match tokenizer training!
SPECIAL_TOKENS = {
    'UNK': 0,  # Unknown token
    'BOS': 1,  # Beginning of sequence
    'EOS': 2,  # End of sequence
    'PAD': 3,  # Padding token
}

TRAIN_CONFIG = {
    'batch_size': 2,
    'lr': 5e-4,
    'epochs': 10,
    'clip': 1.0,
    'ema_decay': 0.999,
    'val_split': 0.1,
    'checkpoint_path': 'utils/checkpoint.pt',
    'warmup_steps': 50,
    'patience': 5,
    'gradient_accumulation_steps': 2,
    'log_interval': 5,
}

INFER_CONFIG = {
    'max_response_length': 100,
    'temperature': 0.8,
    'top_p': 0.9,
    'top_k': 40,
}

# Print config info
if __name__ == "__main__":
    print("="*60)
    print("📋 MODEL CONFIGURATION")
    print("="*60)
    print(f"Vocabulary size: {MODEL_CONFIG['vocab_size']}")
    print(f"Hidden size: {MODEL_CONFIG['hidden_size']}")
    print(f"Layers: {MODEL_CONFIG['num_layers']}")
    print(f"Attention heads: {MODEL_CONFIG['num_heads']}")
    print(f"Sequence length: {MODEL_CONFIG['seq_len']}")
    print(f"MoE experts: {MODEL_CONFIG['num_experts']}")
    
    print("\n" + "="*60)
    print("🎯 TRAINING CONFIGURATION")
    print("="*60)
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"Gradient accumulation: {TRAIN_CONFIG['gradient_accumulation_steps']}")
    print(f"Effective batch size: {TRAIN_CONFIG['batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps']}")
    print(f"Learning rate: {TRAIN_CONFIG['lr']}")
    print(f"Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"Warmup steps: {TRAIN_CONFIG['warmup_steps']}")
    
    print("\n" + "="*60)
    print("🔐 SPECIAL TOKENS")
    print("="*60)
    for token_name, token_id in SPECIAL_TOKENS.items():
        print(f"{token_name}: {token_id}")