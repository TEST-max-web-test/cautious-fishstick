import os
import sys

# ✅ FIXED: Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer

if __name__ == "__main__":
    # Train tokenizer on corpus.txt with PROPER vocab size
    tokenizer = CustomTokenizer()
    
    # ✅ FIXED: Use absolute path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(script_dir, '../data/corpus.txt')
    
    # Verify corpus exists
    if not os.path.exists(corpus_path):
        print(f"❌ ERROR: Corpus file not found at {corpus_path}")
        print(f"   Absolute path: {os.path.abspath(corpus_path)}")
        print("\nPlease ensure corpus.txt exists in ai_cybersec_custom/data/")
        sys.exit(1)
    
    print(f"✅ Found corpus at: {corpus_path}")
    print(f"   File size: {os.path.getsize(corpus_path)} bytes")
    
    # Train tokenizer
    output_prefix = os.path.join(script_dir, 'bpe')
    
    print(f"\n🚀 Training tokenizer...")
    print(f"   Input: {corpus_path}")
    print(f"   Output: {output_prefix}.model")
    print(f"   Vocab size: 8000")
    
    tokenizer.train(
        corpus_path, 
        output_prefix, 
        vocab_size=8000,
        model_type='bpe'
    )
    
    print(f"\n✅ Tokenizer trained and saved as {output_prefix}.model")
    print(f"   Vocabulary size: {tokenizer.vocab_size()} tokens")
    
    # Test the tokenizer
    test_text = "User: What is SQL injection?"
    encoded = tokenizer.encode(test_text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded)
    
    print(f"\n📝 Test encoding:")
    print(f"   Original: {test_text}")
    print(f"   Token count: {len(encoded)}")
    print(f"   Tokens: {encoded[:20]}..." if len(encoded) > 20 else f"   Tokens: {encoded}")
    print(f"   Decoded: {decoded}")
    print(f"\n✅ Tokenizer is working correctly!")