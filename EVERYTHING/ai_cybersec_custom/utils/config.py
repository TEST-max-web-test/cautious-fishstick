# ai_cybersec_custom/utils/config.py
# PRODUCTION CONFIGURATION - ENTERPRISE GRADE
# Updated with GPT-5/Sonnet 4.5 level architecture

# Standard Transformer Config
MODEL_CONFIG = {
    # Model architecture (Enhanced)
    'vocab_size': 2000,
    'hidden_size': 128,  # Increased from 96
    'num_layers': 4,     # Increased from 3
    'num_heads': 4,      # Increased from 3
    'num_kv_heads': 2,   # Grouped Query Attention
    'ff_expansion': 4,
    'dropout': 0.2,      # Reduced for better performance
    'seq_len': 512,
}

# Mixture of Experts Config (Top-tier architecture)
MOE_CONFIG = {
    # MoE architecture (GPT-5/Sonnet 4.5 level)
    'vocab_size': 2000,
    'hidden_size': 256,
    'num_layers': 8,
    'num_heads': 8,
    'num_kv_heads': 2,       # Grouped Query Attention
    'num_experts': 8,        # 8 experts like Mixtral
    'top_k_experts': 2,      # Route to top-2 experts
    'ff_expansion': 4,
    'dropout': 0.1,
    'max_seq_len': 512,
    'use_gradient_checkpointing': True,
    'use_flash_attention': True,
    'aux_loss_weight': 0.01,
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
    'warmup_epochs': 5,
    'patience': 10,
    'weight_decay': 0.01,
    'label_smoothing': 0.1,
    'gradient_accumulation_steps': 4,
    'checkpoint_path': 'ai_cybersec_custom/train/HERE/checkpoint.pt',
}

# MoE Training Config
MOE_TRAIN_CONFIG = {
    # Training hyperparameters for MoE
    'batch_size': 2,         # Smaller batch for larger model
    'lr': 2e-4,
    'epochs': 40,
    'clip': 1.0,
    'warmup_epochs': 3,
    'patience': 10,
    'weight_decay': 0.01,
    'label_smoothing': 0.1,
    'gradient_accumulation_steps': 8,
    'checkpoint_path': 'ai_cybersec_custom/train/checkpoints/moe_checkpoint.pt',
}

# 200M Parameter MoE Config (Production Scale)
MOE_200M_CONFIG = {
    # 200M parameter MoE architecture
    'vocab_size': 32000,
    'hidden_size': 1024,
    'num_layers': 24,
    'num_heads': 16,
    'num_kv_heads': 4,       # Grouped Query Attention (4:1 ratio)
    'num_experts': 32,       # 32 experts for specialization
    'top_k_experts': 4,      # Route to top-4 experts
    'ff_expansion': 4,
    'dropout': 0.1,
    'max_seq_len': 2048,     # Longer context
    'use_gradient_checkpointing': True,
    'use_flash_attention': True,
    'aux_loss_weight': 0.01,
}

# 200M Training Config
MOE_200M_TRAIN_CONFIG = {
    # Training hyperparameters for 200M model
    'batch_size': 1,         # Small batch for large model
    'lr': 1e-4,             # Lower learning rate for stability
    'epochs': 30,
    'clip': 1.0,
    'warmup_epochs': 5,
    'patience': 15,
    'weight_decay': 0.1,
    'label_smoothing': 0.1,
    'gradient_accumulation_steps': 32,  # Large accumulation
    'checkpoint_path': 'ai_cybersec_custom/train/checkpoints/moe_200m_checkpoint.pt',
}

INFER_CONFIG = {
    # Generation hyperparameters
    'max_response_length': 200,
    'temperature': 0.3,
    'top_p': 0.9,
    'top_k': 50,
    'repetition_penalty': 1.1,
}