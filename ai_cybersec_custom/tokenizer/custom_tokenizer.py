import sentencepiece as spm
import os
from typing import List, Optional


class CustomTokenizer:
    """
    Custom tokenizer using SentencePiece with special token support.
    Handles BOS, EOS, and PAD tokens for proper sequence boundaries.
    
    ✅ FIXED: Special tokens now have UNIQUE IDs
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize tokenizer and optionally load a pre-trained model."""
        self.sp = spm.SentencePieceProcessor()
        if model_path and os.path.exists(model_path):
            self.sp.load(model_path)
            # Set special token IDs from loaded model
            self.PAD = self.sp.pad_id()
            self.BOS = self.sp.bos_id()
            self.EOS = self.sp.eos_id()
            self.UNK = self.sp.unk_id()
        else:
            # Default values (will be overwritten after training)
            self.UNK = 0   # UNK = 0 (reserved)
            self.BOS = 1   # BOS = 1
            self.EOS = 2   # EOS = 2
            self.PAD = 3   # PAD = 3 (UNIQUE!)

    def train(self, input_file: str, model_prefix: str, vocab_size: int = 8000, model_type: str = 'bpe'):
        """Train a new SentencePiece tokenizer on the input file.
        
        ✅ FIXED: Special tokens now have UNIQUE IDs to avoid conflicts
        """
        # Train with proper special token configuration
        # IMPORTANT: Each special token must have a UNIQUE ID
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            unk_id=0,          # ✅ UNK = 0 (reserved)
            bos_id=1,          # ✅ BOS = 1
            eos_id=2,          # ✅ EOS = 2
            pad_id=3,          # ✅ PAD = 3 (UNIQUE - was causing conflict!)
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            user_defined_symbols=[],
        )
        
        # Load the trained model
        self.sp.load(f"{model_prefix}.model")
        
        # Update special token IDs
        self.UNK = self.sp.unk_id()
        self.BOS = self.sp.bos_id()
        self.EOS = self.sp.eos_id()
        self.PAD = self.sp.pad_id()
        
        print(f"✅ Tokenizer trained successfully!")
        print(f"   Vocab size: {self.sp.vocab_size()}")
        print(f"   Special tokens: UNK={self.UNK}, BOS={self.BOS}, EOS={self.EOS}, PAD={self.PAD}")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs with optional BOS/EOS tokens."""
        ids = self.sp.encode(text)
        if add_bos:
            ids = [self.BOS] + ids
        if add_eos:
            ids = ids + [self.EOS]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text, removing special tokens."""
        # Filter out special tokens
        ids = [i for i in ids if i not in (self.BOS, self.EOS, self.PAD, self.UNK)]
        return self.sp.decode(ids)
    
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.sp.vocab_size()

    def save(self, path: str):
        """Save tokenizer model to path."""
        if os.path.exists(f"{path}.model"):
            os.rename(f"{path}.model", path)


# Example usage:
if __name__ == "__main__":
    tokenizer = CustomTokenizer()
    tokenizer.train('data/corpus.txt', 'tokenizer/bpe', vocab_size=8000)
    
    # Test encoding
    test_text = "User: What is SQL injection?\nAgent: SQL injection is a security vulnerability."
    ids = tokenizer.encode(test_text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(ids)
    
    print(f"\n📝 Test:")
    print(f"Original: {test_text[:80]}...")
    print(f"Token IDs (first 20): {ids[:20]}")
    print(f"Max token ID: {max(ids)} (should be < {tokenizer.vocab_size()})")
    print(f"Decoded: {decoded[:80]}...")