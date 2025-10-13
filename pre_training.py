#!/usr/bin/env python3
"""
Pre-Training Verification Script
Run this BEFORE training to ensure everything is ready
"""
import os
import sys

# Add to path
sys.path.insert(0, '.')

def check(condition, success_msg, fail_msg):
    """Helper to check conditions and print results"""
    if condition:
        print(f"✅ {success_msg}")
        return True
    else:
        print(f"❌ {fail_msg}")
        return False

print("="*70)
print("🔍 PRE-TRAINING VERIFICATION")
print("="*70)
print()

all_good = True

# ============================================================================
# 1. Check Directory Structure
# ============================================================================
print("📁 Checking directory structure...")
all_good &= check(
    os.path.exists('ai_cybersec_custom'),
    "Found ai_cybersec_custom directory",
    "Missing ai_cybersec_custom directory!"
)
all_good &= check(
    os.path.exists('ai_cybersec_custom/model'),
    "Found model directory",
    "Missing ai_cybersec_custom/model directory!"
)
all_good &= check(
    os.path.exists('ai_cybersec_custom/data'),
    "Found data directory",
    "Missing ai_cybersec_custom/data directory!"
)
print()

# ============================================================================
# 2. Check Required Files Exist
# ============================================================================
print("📄 Checking required files...")

required_files = {
    'ai_cybersec_custom/model/custom_transformer.py': 'Model file',
    'ai_cybersec_custom/utils/config.py': 'Config file',
    'ai_cybersec_custom/train/train.py': 'Training script',
    'ai_cybersec_custom/chat.py': 'Chat interface',
    'ai_cybersec_custom/data/corpus.txt': 'Training corpus',
}

for filepath, description in required_files.items():
    all_good &= check(
        os.path.exists(filepath),
        f"Found {description}: {filepath}",
        f"Missing {description}: {filepath}"
    )
print()

# ============================================================================
# 3. Check Model File (Advanced vs Old)
# ============================================================================
print("🧠 Checking model architecture...")
try:
    with open('ai_cybersec_custom/model/custom_transformer.py', 'r') as f:
        content = f.read()
    
    has_rope = 'RotaryPositionalEmbedding' in content or 'RoPE' in content
    has_gated = 'GatedFFN' in content or 'gate_proj' in content
    has_reasoning = 'ReasoningTransformer' in content or 'ReasoningBlock' in content
    
    if has_rope and has_gated:
        print("✅ Model file contains Advanced Reasoning architecture")
        print("   • Rotary Position Embeddings: ✓")
        print("   • Gated Feed-Forward: ✓")
        if has_reasoning:
            print("   • Reasoning blocks: ✓")
    else:
        print("❌ Model file appears to be OLD version!")
        print("   Please replace with 'Advanced Reasoning Transformer' artifact")
        all_good = False
except Exception as e:
    print(f"❌ Could not read model file: {e}")
    all_good = False
print()

# ============================================================================
# 4. Check Config File
# ============================================================================
print("⚙️  Checking configuration...")
try:
    from ai_cybersec_custom.utils.config import MODEL_CONFIG, TRAIN_CONFIG, INFER_CONFIG
    
    # Check if it's the new config
    has_dropout = 'dropout' in MODEL_CONFIG
    has_label_smoothing = 'label_smoothing' in TRAIN_CONFIG
    has_rep_penalty = 'repetition_penalty' in INFER_CONFIG
    
    if has_dropout and has_label_smoothing and has_rep_penalty:
        print("✅ Config file is NEW version (Reasoning Config)")
        print(f"   • Model: {MODEL_CONFIG['num_layers']}L x {MODEL_CONFIG['hidden_size']}D")
        print(f"   • Dropout: {MODEL_CONFIG.get('dropout', 'N/A')}")
        print(f"   • Label smoothing: {TRAIN_CONFIG.get('label_smoothing', 'N/A')}")
        print(f"   • Repetition penalty: {INFER_CONFIG.get('repetition_penalty', 'N/A')}")
    else:
        print("❌ Config file appears to be OLD version!")
        print("   Please replace with 'Config for Reasoning Model' artifact")
        all_good = False
        
except Exception as e:
    print(f"❌ Could not load config: {e}")
    all_good = False
print()

# ============================================================================
# 5. Check Tokenizer
# ============================================================================
print("📝 Checking tokenizer...")
tokenizer_path = 'ai_cybersec_custom/tokenizer/bpe.model'

if os.path.exists(tokenizer_path):
    print(f"✅ Tokenizer exists: {tokenizer_path}")
    
    try:
        from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
        tokenizer = CustomTokenizer(tokenizer_path)
        
        vocab_size = tokenizer.vocab_size()
        print(f"   Vocab size: {vocab_size}")
        
        # Check if matches config
        from ai_cybersec_custom.utils.config import MODEL_CONFIG
        if vocab_size == MODEL_CONFIG['vocab_size']:
            print(f"   ✅ Vocab size matches config ({MODEL_CONFIG['vocab_size']})")
        else:
            print(f"   ❌ Vocab size mismatch! Tokenizer: {vocab_size}, Config: {MODEL_CONFIG['vocab_size']}")
            print("   Run: python3 verify_tokenizer.py for details")
            all_good = False
            
    except Exception as e:
        print(f"   ⚠️  Could not verify tokenizer: {e}")
else:
    print(f"⚠️  Tokenizer not found: {tokenizer_path}")
    print("   Will be created during training")
print()

# ============================================================================
# 6. Check Corpus
# ============================================================================
print("📚 Checking training corpus...")
corpus_path = 'ai_cybersec_custom/data/corpus.txt'

if os.path.exists(corpus_path):
    with open(corpus_path, 'r') as f:
        content = f.read()
    
    lines = [line for line in content.split('\n') if line.strip()]
    size_kb = len(content) / 1024
    
    print(f"✅ Corpus found: {len(lines)} lines, {size_kb:.1f} KB")
    
    if len(lines) < 100:
        print(f"   ⚠️  Corpus is small ({len(lines)} lines). Recommend 600+")
    elif len(lines) < 600:
        print(f"   ✓ Corpus size acceptable ({len(lines)} lines)")
    else:
        print(f"   ✓ Corpus size good ({len(lines)} lines)")
        
    # Estimate samples
    conversations = content.strip().split('\n\n')
    sample_count = len([c for c in conversations if c.strip()])
    print(f"   Estimated training samples: ~{sample_count}")
    
else:
    print(f"❌ Corpus not found: {corpus_path}")
    all_good = False
print()

# ============================================================================
# 7. Check Python Dependencies
# ============================================================================
print("🐍 Checking Python dependencies...")

dependencies = {
    'torch': 'PyTorch',
    'sentencepiece': 'SentencePiece',
    'sklearn': 'scikit-learn',
}

for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"✅ {name} installed")
    except ImportError:
        print(f"❌ {name} NOT installed")
        print(f"   Install with: pip install {module}")
        all_good = False
print()

# ============================================================================
# 8. Check for Old Checkpoints
# ============================================================================
print("💾 Checking for old checkpoints...")

checkpoint_paths = [
    'ai_cybersec_custom/train/utils/checkpoint.pt',
    'ai_cybersec_custom/utils/checkpoint.pt',
    'utils/checkpoint.pt',
]

old_found = False
for path in checkpoint_paths:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"⚠️  Found old checkpoint: {path} ({size:.1f} MB)")
        old_found = True

if old_found:
    print("   ⚠️  Old checkpoints should be removed before training")
    print("   Run: rm -f ai_cybersec_custom/train/utils/checkpoint.pt")
else:
    print("✅ No old checkpoints found")
print()

# ============================================================================
# 9. Estimate Resources
# ============================================================================
print("💻 Resource estimates...")
try:
    from ai_cybersec_custom.utils.config import MODEL_CONFIG, TRAIN_CONFIG
    
    # Rough parameter estimate
    params = (
        MODEL_CONFIG['vocab_size'] * MODEL_CONFIG['hidden_size'] +
        MODEL_CONFIG['num_layers'] * (
            4 * MODEL_CONFIG['hidden_size']**2 +
            3 * MODEL_CONFIG['hidden_size'] * MODEL_CONFIG['hidden_size'] * MODEL_CONFIG['ff_expansion']
        )
    )
    
    print(f"Model size: ~{params//1000}K parameters")
    print(f"Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
    print()
    print("Estimated training time:")
    print("  • CPU (4 cores): 2-5 hours")
    print("  • GPU (modern): 15-45 minutes")
    
except Exception as e:
    print(f"Could not estimate: {e}")
print()

# ============================================================================
# Final Summary
# ============================================================================
print("="*70)
if all_good:
    print("✅ ALL CHECKS PASSED - READY TO TRAIN!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. cd ai_cybersec_custom/train")
    print("  2. python3 train.py")
    print("  3. Wait for training to complete (watch for PPL < 5)")
    print("  4. python3 ../../ai_cybersec_custom/chat.py")
else:
    print("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE TRAINING")
    print("="*70)
    print()
    print("Review the errors above and:")
    print("  1. Replace required files with artifacts")
    print("  2. Install missing dependencies")
    print("  3. Fix configuration mismatches")
    print("  4. Re-run this script")

print()
sys.exit(0 if all_good else 1)