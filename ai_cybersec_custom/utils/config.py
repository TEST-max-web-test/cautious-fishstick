# Centralized configuration for model, training, and inference
# FIXED VERSION with proper vocab size and settings

# Model Presets
TINY_MODEL = {
    'vocab_size': 8000,  # ✅ FIXED: Matches new tokenizer
    'hidden_size': 128,
    'num_layers': 2,
    'num_heads': 4,
    'ff_expansion': 2,
    'num_experts': 2,
    'moe_k': 1,
    'seq_len': 512,  # ✅ FIXED: Longer sequences for conversations
}

SMALL_MODEL = {
    'vocab_size': 8000,  # ✅ FIXED
    'hidden_size': 256,
    'num_layers': 4,
    'num_heads': 8,
    'ff_expansion': 2,
    'num_experts': 4,
    'moe_k': 2,
    'seq_len': 512,
}

MEDIUM_MODEL = {
    'vocab_size': 8000,  # ✅ FIXED
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

TRAIN_CONFIG = {
    'batch_size': 2,  # ✅ Increased slightly
    'lr': 5e-4,  # ✅ Better learning rate
    'epochs': 10,  # ✅ More epochs for proper training
    'clip': 1.0,
    'ema_decay': 0.999,
    'val_split': 0.1,
    'checkpoint_path': 'utils/checkpoint.pt',
    'warmup_steps': 50,  # ✅ Proper warmup
    'patience': 5,  # ✅ More patience
    'gradient_accumulation_steps': 2,  # ✅ Effective batch size = 4
    'log_interval': 5,  # Log every 5 batches
}

INFER_CONFIG = {
    'max_response_length': 100,  # ✅ Longer responses
    'temperature': 0.8,  # ✅ Less random
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