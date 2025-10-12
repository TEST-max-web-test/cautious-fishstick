#!/usr/bin/env python3
"""
Complete setup and training script for AI Cybersec Custom
Place this file in the PROJECT ROOT (same level as ai_cybersec_custom folder)
Run with: python3 setup_and_train.py
"""

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"❌ Failed: {description}")
            print(f"Error: {result.stderr}")
            return False
        print(f"✅ {description} - Done")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_file_exists(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f"✅ {description} found at: {path}")
        return True
    else:
        print(f"❌ {description} NOT found at: {path}")
        return False

def main():
    print_header("AI CYBERSEC CUSTOM - COMPLETE SETUP")
    
    # 1. Check we're in the right directory
    if not os.path.exists('ai_cybersec_custom'):
        print("❌ ERROR: 'ai_cybersec_custom' folder not found")
        print(f"   Current directory: {os.getcwd()}")
        print("   Please run this script from the project root")
        sys.exit(1)
    
    print(f"📁 Working directory: {os.getcwd()}")
    
    # 2. Check Python version
    print_header("CHECKING PYTHON")
    print(f"🐍 Python version: {sys.version}")
    
    # 3. Install dependencies
    print_header("INSTALLING DEPENDENCIES")
    deps = ["torch", "sentencepiece", "scikit-learn"]
    
    for dep in deps:
        try:
            __import__(dep.replace('-', '_'))
            print(f"✅ {dep} already installed")
        except ImportError:
            print(f"📦 Installing {dep}...")
            if not run_command(f"pip install {dep}", f"Installing {dep}"):
                print(f"⚠️  Warning: Failed to install {dep}")
    
    # 4. Check corpus file
    print_header("CHECKING DATA FILES")
    corpus_path = "ai_cybersec_custom/data/corpus.txt"
    if not check_file_exists(corpus_path, "Corpus file"):
        print("\n❌ CRITICAL: corpus.txt is required for training")
        sys.exit(1)
    
    corpus_size = os.path.getsize(corpus_path)
    print(f"   Corpus size: {corpus_size:,} bytes ({corpus_size/1024:.1f} KB)")
    
    # 5. Check/Train tokenizer
    print_header("TOKENIZER SETUP")
    tokenizer_path = "ai_cybersec_custom/tokenizer/bpe.model"
    
    if os.path.exists(tokenizer_path):
        print(f"✅ Tokenizer already exists at: {tokenizer_path}")
        print("   Skipping tokenizer training")
    else:
        print(f"⚠️  Tokenizer not found, training now...")
        
        # Add to path and import
        sys.path.insert(0, os.path.abspath('.'))
        
        try:
            from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
            
            print("🚀 Training tokenizer...")
            tokenizer = CustomTokenizer()
            tokenizer.train(
                corpus_path,
                'ai_cybersec_custom/tokenizer/bpe',
                vocab_size=8000,
                model_type='bpe'
            )
            print("✅ Tokenizer training complete!")
            
        except Exception as e:
            print(f"❌ Tokenizer training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # 6. Verify tokenizer works
    print("\n🔍 Verifying tokenizer...")
    try:
        sys.path.insert(0, os.path.abspath('.'))
        from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
        
        tokenizer = CustomTokenizer(tokenizer_path)
        test_text = "User: What is SQL injection?"
        ids = tokenizer.encode(test_text)
        
        print(f"✅ Tokenizer verified")
        print(f"   Vocab size: {tokenizer.vocab_size()}")
        print(f"   Test tokens: {len(ids)}")
        print(f"   Special tokens: PAD={tokenizer.PAD}, UNK={tokenizer.UNK}, BOS={tokenizer.BOS}, EOS={tokenizer.EOS}")
        
    except Exception as e:
        print(f"❌ Tokenizer verification failed: {e}")
        sys.exit(1)
    
    # 7. Create directories
    print_header("CREATING DIRECTORIES")
    os.makedirs('ai_cybersec_custom/utils', exist_ok=True)
    print("✅ Created ai_cybersec_custom/utils/")
    
    # 8. Start training
    print_header("READY TO TRAIN")
    print("✅ All setup steps completed!")
    print("\n📋 To start training, run:")
    print("   cd ai_cybersec_custom/train")
    print("   python3 train.py")
    print("\nOr run this one-liner from project root:")
    print("   cd ai_cybersec_custom/train && python3 train.py")
    
    # 9. Success!
    print_header("SETUP COMPLETE!")
    print("\n✅ All steps completed successfully!")
    print("\nNext steps:")
    print("  1. Check checkpoint: ai_cybersec_custom/utils/checkpoint.pt")
    print("  2. Run chat: python3 ai_cybersec_custom/chat.py")
    print("  3. Evaluate: python3 ai_cybersec_custom/eval/evaluate.py")
    print("")

if __name__ == "__main__":
    main()