# Optimized configuration for SMALL dataset training
# This will actually work with 150-500 samples

# Ultra-tiny model for small datasets
ULTRA_TINY_MODEL = {
    'vocab_size': 2000,  # ✅ Much smaller vocab for limited data
    'hidden_size': 64,   # ✅ Tiny hidden size
    'num_layers': 2,     # ✅ Only 2 layers
    'num_heads': 4,      # ✅ Fewer heads
    'ff_expansion': 2,   # ✅ Smaller feedforward
    'num_experts': 1,    # ✅ NO MoE (too complex for small data)
    'moe_k': 1,
    'seq_len': 256,      # ✅ Shorter sequences
}

TINY_MODEL = {
    'vocab_size': 2000,
    'hidden_size': 128,
    'num_layers': 3,
    'num_heads': 4,
    'ff_expansion': 2,
    'num_experts': 2,
    'moe_k': 1,
    'seq_len': 256,
}

SMALL_MODEL = {
    'vocab_size': 2000,  # ✅ Reduced from 8000
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

# ✅ USE ULTRA_TINY for datasets < 1000 samples
# ✅ USE TINY for datasets 1000-5000 samples  
# ✅ USE SMALL for datasets 5000-50000 samples
MODEL_CONFIG = ULTRA_TINY_MODEL  # ✅ CHANGED: Was SMALL_MODEL

SPECIAL_TOKENS = {
    'PAD': 0,
    'UNK': 1,
    'BOS': 2,
    'EOS': 3,
}

TRAIN_CONFIG = {
    'batch_size': 4,      # ✅ Increased from 2
    'lr': 1e-3,           # ✅ Higher learning rate for faster learning
    'epochs': 100,        # ✅ INCREASED from 10 (need much more)
    'clip': 1.0,
    'ema_decay': 0.999,
    'val_split': 0.1,
    'checkpoint_path': 'utils/checkpoint.pt',
    'warmup_steps': 20,   # ✅ Reduced warmup
    'patience': 20,       # ✅ More patience
    'gradient_accumulation_steps': 1,  # ✅ No accumulation for small model
    'log_interval': 5,
}

INFER_CONFIG = {
    'max_response_length': 50,   # ✅ Shorter responses
    'temperature': 1.0,          # ✅ Higher temp for more variety
    'top_p': 0.95,
    'top_k': 20,                 # ✅ Lower for more focused sampling
}

# Print config info
if __name__ == "__main__":
    print("="*60)
    print("📋 OPTIMIZED MODEL CONFIGURATION")
    print("="*60)
    print(f"Model: ULTRA_TINY (for small datasets)")
    print(f"Vocabulary size: {MODEL_CONFIG['vocab_size']}")
    print(f"Hidden size: {MODEL_CONFIG['hidden_size']}")
    print(f"Layers: {MODEL_CONFIG['num_layers']}")
    print(f"Attention heads: {MODEL_CONFIG['num_heads']}")
    print(f"Sequence length: {MODEL_CONFIG['seq_len']}")
    print(f"\nParameters estimate: ~100K (vs 8M before)")
    print("\n" + "="*60)
    print("🎯 OPTIMIZED TRAINING CONFIGURATION")
    print("="*60)
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"Learning rate: {TRAIN_CONFIG['lr']}")
    print(f"Epochs: {TRAIN_CONFIG['epochs']} (was 10)")
    print(f"Patience: {TRAIN_CONFIG['patience']}")
    print("\n💡 This config is optimized for 150-500 training samples")
    print("   For better results, expand your corpus to 1000+ samples")