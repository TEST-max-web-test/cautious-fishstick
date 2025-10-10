import os
from custom_tokenizer import CustomTokenizer

if __name__ == "__main__":
    # Train tokenizer on corpus.txt with PROPER vocab size
    tokenizer = CustomTokenizer()
    
    corpus_path = os.path.join(os.path.dirname(__file__), '../data/corpus.txt')
    
    # ✅ FIXED: Use proper vocab size (8000 is good for specialized domain)
    tokenizer.train(
        corpus_path, 
        'bpe', 
        vocab_size=8000,  # ✅ Much larger vocabulary
        model_type='bpe'
    )
    
    print("✅ Tokenizer trained and saved as bpe.model")
    print(f"   Vocabulary size: 8000 tokens")
    
    # Test the tokenizer
    test_text = "User: What is SQL injection?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\n📝 Test encoding:")
    print(f"   Original: {test_text}")
    print(f"   Tokens: {encoded[:20]}...")  # Show first 20 tokens
    print(f"   Decoded: {decoded}")