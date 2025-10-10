import sentencepiece as spm
import os
from typing import List, Optional


class CustomTokenizer:
    """
    Custom tokenizer using SentencePiece with special token support.
    Handles BOS, EOS, and PAD tokens for proper sequence boundaries.
    """
    BOS = 50256
    EOS = 50257
    PAD = 0

    def __init__(self, model_path: Optional[str] = None):
        """Initialize tokenizer and optionally load a pre-trained model."""
        self.sp = spm.SentencePieceProcessor()
        if model_path:
            self.sp.load(model_path)

    def train(self, input_file: str, model_prefix: str, vocab_size: int = 50258, model_type: str = 'bpe'):
        """Train a new SentencePiece tokenizer on the input file."""
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            unk_id=self.PAD,
            bos_id=self.BOS,
            eos_id=self.EOS
        )
        self.sp.load(f"{model_prefix}.model")

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
        ids = [i for i in ids if i not in (self.BOS, self.EOS, self.PAD)]
        return self.sp.decode(ids)

    def save(self, path: str):
        """Save tokenizer model to path."""
        os.rename(f"{path}.model", path)


# Example usage:
# tokenizer = CustomTokenizer()
# tokenizer.train('data/corpus.txt', 'tokenizer/bpe', vocab_size=50258)
# ids = tokenizer.encode('Hello world!')
# print(tokenizer.decode(ids))