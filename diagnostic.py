#!/usr/bin/env python3
"""
DIAGNOSTIC SCRIPT
Run this if something isn't working: python3 diagnostic.py
"""
import os
import sys

print("="*70)
print("🔍 DIAGNOSTIC CHECK")
print("="*70)
print()

errors = []
warnings = []

# 1. Check Python version
print("1. Python version...")
if sys.version_info < (3, 7):
    errors.append(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old. Need 3.7+")
else:
    print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}")

# 2. Check packages
print("\n2. Checking packages...")
packages = {
    'torch': 'PyTorch',
    'sentencepiece': 'SentencePiece',
    'sklearn': 'scikit-learn'
}

for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f"   ✅ {name}")
    except ImportError:
        errors.append(f"{name} not installed. Run: pip install {pkg}")

# 3. Check directory structure
print("\n3. Checking directory structure...")
required_dirs = [
    'ai_cybersec_custom',
    'ai_cybersec_custom/model',
    'ai_cybersec_custom/data',
    'ai_cybersec_custom/tokenizer',
    'ai_cybersec_custom/train',
    'ai_cybersec_custom/utils',
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"   ✅ {dir_path}")
    else:
        errors.append(f"Missing directory: {dir_path}")

# 4. Check required files
print("\n4. Checking required files...")
required_files = [
    ('ai_cybersec_custom/data/corpus.txt', 'Training data'),
    ('ai_cybersec_custom/model/custom_transformer.py', 'Model'),
    ('ai_cybersec_custom/utils/config.py', 'Config'),
    ('ai_cybersec_custom/train/train.py', 'Training script'),
    ('ai_cybersec_custom/tokenizer/custom_tokenizer.py', 'Tokenizer class'),
    ('ai_cybersec_custom/tokenizer/train_tokenizer.py', 'Tokenizer trainer'),
    ('ai_cybersec_custom/data/text_dataset.py', 'Dataset loader'),
    ('ai_cybersec_custom/chat.py', 'Chat interface'),
]

for file_path, description in required_files:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"   ✅ {description}: {size:,} bytes")
    else:
        errors.append(f"Missing file: {file_path} ({description})")

# 5. Check corpus
print("\n5. Checking corpus...")
corpus_path = 'ai_cybersec_custom/data/corpus.txt'
if os.path.exists(corpus_path):
    with open(corpus_path, 'r') as f:
        lines = f.readlines()
    print(f"   ✅ Corpus has {len(lines)} lines")
    
    if len(lines) < 100:
        warnings.append(f"Corpus is small ({len(lines)} lines). Recommend 600+")
    
    # Check format
    user_count = sum(1 for line in lines if line.strip().startswith('User:'))
    agent_count = sum(1 for line in lines if line.strip().startswith('Agent:'))
    print(f"   ✅ User lines: {user_count}, Agent lines: {agent_count}")
    
    if abs(user_count - agent_count) > 5:
        warnings.append("User/Agent lines don't match. Check corpus format")
else:
    errors.append("Corpus file not found!")

# 6. Check tokenizer
print("\n6. Checking tokenizer...")
tokenizer_path = 'ai_cybersec_custom/tokenizer/bpe.model'
if os.path.exists(tokenizer_path):
    print(f"   ✅ Tokenizer trained")
else:
    warnings.append("Tokenizer not trained yet. Run: cd ai_cybersec_custom/tokenizer && python3 train_tokenizer.py")

# 7. Check checkpoint
print("\n7. Checking model checkpoint...")
checkpoint_paths = [
    'ai_cybersec_custom/train/utils/checkpoint.pt',
    'ai_cybersec_custom/utils/checkpoint.pt',
]
found = False
for path in checkpoint_paths:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✅ Checkpoint found: {path} ({size:.1f} MB)")
        found = True
        break

if not found:
    warnings.append("No checkpoint found. Model not trained yet. Run: cd ai_cybersec_custom/train && python3 train.py")

# 8. Test imports
print("\n8. Testing imports...")
sys.path.insert(0, '.')

try:
    from ai_cybersec_custom.utils.config import MODEL_CONFIG
    print(f"   ✅ Config loads: {MODEL_CONFIG['num_layers']}L x {MODEL_CONFIG['hidden_size']}D")
except Exception as e:
    errors.append(f"Config import failed: {e}")

try:
    from ai_cybersec_custom.model.custom_transformer import CustomTransformer
    print(f"   ✅ Model class loads")
except Exception as e:
    errors.append(f"Model import failed: {e}")

try:
    from ai_cybersec_custom.data.text_dataset import TextDataset
    print(f"   ✅ Dataset class loads")
except Exception as e:
    errors.append(f"Dataset import failed: {e}")

# Summary
print("\n" + "="*70)
if errors:
    print("❌ ERRORS FOUND:")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")
else:
    print("✅ NO ERRORS")

if warnings:
    print("\n⚠️  WARNINGS:")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")

print("="*70)

if not errors:
    print("\n✅ Everything looks good!")
    if warnings:
        print("⚠️  You have warnings, but can proceed.")
    print("\nNext step:")
    if 'Tokenizer not trained' in str(warnings):
        print("   cd ai_cybersec_custom/tokenizer")
        print("   python3 train_tokenizer.py")
    elif 'No checkpoint' in str(warnings):
        print("   cd ai_cybersec_custom/train")
        print("   python3 train.py")
    else:
        print("   python3 ai_cybersec_custom/chat.py")
else:
    print("\n❌ Fix the errors above, then run this script again.")
    sys.exit(1)