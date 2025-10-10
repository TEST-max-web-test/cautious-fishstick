#!/usr/bin/env python3
"""
Verify that tokenizer and model config match properly
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG

print("="*70)
print("🔍 TOKENIZER & CONFIG VERIFICATION")
print("="*70)

# Load tokenizer
tokenizer_path = 'bpe.model'
if not os.path.exists(tokenizer_path):
    print(f"❌ ERROR: {tokenizer_path} not found!")
    print(f"   Please run: python ai_cybersec_custom/tokenizer/train_tokenizer.py")
    sys.exit(1)

print(f"\n📂 Loading tokenizer from {tokenizer_path}...")
tokenizer = CustomTokenizer(tokenizer_path)

# Get sizes
actual_vocab = tokenizer.vocab_size()
expected_vocab = MODEL_CONFIG['vocab_size']

print(f"\n📊 Vocabulary Size Check:")
print(f"   Tokenizer vocab size: {actual_vocab}")
print(f"   Model config expects: {expected_vocab}")

if actual_vocab != expected_vocab:
    print(f"\n❌ MISMATCH! These must be equal!")
    print(f"\n🔧 Fix options:")
    print(f"   1. Update MODEL_CONFIG['vocab_size'] to {actual_vocab} in config.py")
    print(f"   2. OR retrain tokenizer with vocab_size={expected_vocab}")
    sys.exit(1)
else:
    print(f"   ✅ MATCH! Configuration is correct.")

# Test special tokens
print(f"\n🏷️  Special Tokens:")
print(f"   PAD: {tokenizer.PAD}")
print(f"   BOS: {tokenizer.BOS}")
print(f"   EOS: {tokenizer.EOS}")
print(f"   UNK: {tokenizer.UNK}")

# Verify special tokens are within vocab
max_special = max(tokenizer.PAD, tokenizer.BOS, tokenizer.EOS, tokenizer.UNK)
if max_special >= actual_vocab:
    print(f"\n❌ ERROR: Special token ID {max_special} is outside vocab size {actual_vocab}!")
    print(f"   Special tokens must be < vocab_size")
    print(f"   Please retrain tokenizer with the fixed version.")
    sys.exit(1)
else:
    print(f"   ✅ All special tokens within vocab range (< {actual_vocab})")

# Test encoding
print(f"\n📝 Test Encoding:")
test_text = "User: What is SQL injection?"
ids = tokenizer.encode(test_text, add_bos=True, add_eos=True)
decoded = tokenizer.decode(ids)

print(f"   Original:  {test_text}")
print(f"   Token IDs: {ids[:15]}..." if len(ids) > 15 else f"   Token IDs: {ids}")
print(f"   Max ID:    {max(ids)} (must be < {actual_vocab})")
print(f"   Decoded:   {decoded}")

if max(ids) >= actual_vocab:
    print(f"\n❌ ERROR: Token ID {max(ids)} exceeds vocab size {actual_vocab}!")
    print(f"   This will cause IndexError during training!")
    print(f"   Please retrain tokenizer with the fixed version.")
    sys.exit(1)
else:
    print(f"   ✅ All token IDs within valid range")

# Test with dataset sample
print(f"\n📚 Test with Dataset Sample:")
corpus_path = 'ai_cybersec_custom/data/corpus.txt'
if os.path.exists(corpus_path):
    with open(corpus_path, 'r') as f:
        sample = f.read(500)  # First 500 chars
    
    sample_ids = tokenizer.encode(sample, add_bos=True, add_eos=True)
    print(f"   Sample length: {len(sample_ids)} tokens")
    print(f"   Max token ID:  {max(sample_ids)}")
    
    if max(sample_ids) >= actual_vocab:
        print(f"   ❌ ERROR: Found token ID {max(sample_ids)} >= {actual_vocab}")
        sys.exit(1)
    else:
        print(f"   ✅ All corpus tokens within valid range")

print("\n" + "="*70)
print("✅ ALL CHECKS PASSED! Ready to train.")
print("="*70)
print("\nNext steps:")
print("  1. Run: PYTHONPATH=./ python3 ai_cybersec_custom/train/train.py")
print("  2. Monitor training loss (should decrease)")
print("  3. After training: python3 ai_cybersec_custom/chat.py")