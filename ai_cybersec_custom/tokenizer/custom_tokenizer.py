import sentencepiece as spm
import os

class CustomTokenizer:
    def __init__(self, model_path=None):
        self.sp = spm.SentencePieceProcessor()
        if model_path:
            self.sp.load(model_path)

    def train(self, input_file, model_prefix, vocab_size=32000, model_type='bpe'):
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            unk_id=0
        )
        self.sp.load(f"{model_prefix}.model")

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

    def save(self, path):
        os.rename(f"{path}.model", path)

# Example usage:
# tokenizer = CustomTokenizer()
# tokenizer.train('data/corpus.txt', 'tokenizer/bpe', vocab_size=32000)
# ids = tokenizer.encode('Hello world!')
# print(tokenizer.decode(ids))
