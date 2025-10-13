# Configuration for Advanced Reasoning Transformer

# Small but smart model for ~600 line corpus
REASONING_SMALL = {
    'vocab_size': 2000,
    'hidden_size': 256,     # Larger hidden size for better representations
    'num_layers': 6,        # Deeper for reasoning
    'num_heads': 8,         # More heads for attention
    'ff_expansion': 4,      # Wider FFN for processing
    'dropout': 0.1,         # Prevent overfitting
    'seq_len': 512,
}

# Medium model for better results (if you have GPU)
REASONING_MEDIUM = {
    'vocab_size': 2000,
    'hidden_size': 384,
    'num_layers': 8,
    'num_heads': 8,
    'ff_expansion': 4,
    'dropout': 0.1,
    'seq_len': 512,
}

# Tiny model for CPU/quick testing
REASONING_TINY = {
    'vocab_size': 2000,
    'hidden_size': 128,
    'num_layers': 4,
    'num_heads': 4,
    'ff_expansion': 4,
    'dropout': 0.1,
    'seq_len': 512,
}

# Use SMALL by default (good balance)
MODEL_CONFIG = REASONING_SMALL

SPECIAL_TOKENS = {
    'PAD': 0,
    'UNK': 1,
    'BOS': 2,
    'EOS': 3,
}

TRAIN_CONFIG = {
    'batch_size': 8,           # Larger batches for stable training
    'lr': 3e-4,                # Standard transformer learning rate
    'epochs': 200,             # More epochs needed for reasoning
    'clip': 1.0,               # Gradient clipping
    'ema_decay': 0.999,
    'val_split': 0.1,
    'checkpoint_path': 'ai_cybersec_custom/train/utils/checkpoint.pt',
    'warmup_steps': 100,       # Longer warmup for stability
    'patience': 30,            # More patience for deeper model
    'gradient_accumulation_steps': 2,  # Effective batch size = 16
    'log_interval': 10,
    
    # New: Label smoothing to prevent overconfidence
    'label_smoothing': 0.1,
    
    # New: Weight decay for regularization
    'weight_decay': 0.01,
}

INFER_CONFIG = {
    'max_response_length': 100,    # Longer responses
    'temperature': 0.8,            # Balanced creativity
    'top_p': 0.92,                 # Nucleus sampling
    'top_k': 50,                   # Consider top 50 tokens
    
    # New: Repetition penalty to avoid loops
    'repetition_penalty': 1.2,
}

# Print config info
if __name__ == "__main__":
    print("="*70)
    print("🧠 ADVANCED REASONING MODEL CONFIGURATION")
    print("="*70)
    print(f"\nModel: REASONING_SMALL")
    print(f"  Vocabulary size: {MODEL_CONFIG['vocab_size']}")
    print(f"  Hidden size: {MODEL_CONFIG['hidden_size']}")
    print(f"  Layers: {MODEL_CONFIG['num_layers']}")
    print(f"  Attention heads: {MODEL_CONFIG['num_heads']}")
    print(f"  FF expansion: {MODEL_CONFIG['ff_expansion']}")
    print(f"  Dropout: {MODEL_CONFIG['dropout']}")
    print(f"  Sequence length: {MODEL_CONFIG['seq_len']}")
    
    # Estimate parameters
    params = (
        MODEL_CONFIG['vocab_size'] * MODEL_CONFIG['hidden_size'] +  # Embeddings
        MODEL_CONFIG['num_layers'] * (
            4 * MODEL_CONFIG['hidden_size']**2 +  # Attention QKV + output
            3 * MODEL_CONFIG['hidden_size'] * MODEL_CONFIG['hidden_size'] * MODEL_CONFIG['ff_expansion']  # FFN
        )
    )
    print(f"\n  Estimated parameters: ~{params//1000}K")
    
    print(f"\n" + "="*70)
    print("🎯 TRAINING CONFIGURATION")
    print("="*70)
    print(f"  Batch size: {TRAIN_CONFIG['batch_size']} (x{TRAIN_CONFIG['gradient_accumulation_steps']} accum)")
    print(f"  Effective batch: {TRAIN_CONFIG['batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps']}")
    print(f"  Learning rate: {TRAIN_CONFIG['lr']}")
    print(f"  Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"  Warmup steps: {TRAIN_CONFIG['warmup_steps']}")
    print(f"  Patience: {TRAIN_CONFIG['patience']}")
    print(f"  Label smoothing: {TRAIN_CONFIG['label_smoothing']}")
    print(f"  Weight decay: {TRAIN_CONFIG['weight_decay']}")
    
    print(f"\n" + "="*70)
    print("💬 INFERENCE CONFIGURATION")
    print("="*70)
    print(f"  Max response length: {INFER_CONFIG['max_response_length']}")
    print(f"  Temperature: {INFER_CONFIG['temperature']}")
    print(f"  Top-p: {INFER_CONFIG['top_p']}")
    print(f"  Top-k: {INFER_CONFIG['top_k']}")
    print(f"  Repetition penalty: {INFER_CONFIG['repetition_penalty']}")
    
    print("\n" + "="*70)
    print("✨ KEY IMPROVEMENTS")
    print("="*70)
    print("  ✅ Deeper architecture (6 layers vs 2-3)")
    print("  ✅ Better attention with RoPE positional encoding")
    print("  ✅ Gated FFN for selective reasoning")
    print("  ✅ Proper dropout to prevent memorization")
    print("  ✅ Label smoothing for better generalization")
    print("  ✅ Repetition penalty for varied responses")
    print("  ✅ Longer training (200 epochs)")
    
    print("\n" + "="*70)
    print("📈 EXPECTED RESULTS")
    print("="*70)
    print("  ✅ Generate novel sentences (not memorized)")
    print("  ✅ Understand context and reasoning")
    print("  ✅ Generalize to unseen questions")
    print("  ✅ Varied responses (not repetitive)")
    print("  ⏱️  Training time: 20-60 minutes (depending on hardware)")
    print("="*70)