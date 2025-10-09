import sentencepiece as spm
import os

class CustomTokenizer:
    BOS = 50256
    EOS = 50257
    PAD = 0
    def __init__(self, model_path=None):
        self.sp = spm.SentencePieceProcessor()
        if model_path:
            self.sp.load(model_path)

    def train(self, input_file, model_prefix, vocab_size=50258, model_type='bpe'):
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

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.sp.encode(text)
        if add_bos:
            ids = [self.BOS] + ids
        if add_eos:
            ids = ids + [self.EOS]
        return ids

    def decode(self, ids):
        # Remove special tokens
        ids = [i for i in ids if i not in (self.BOS, self.EOS, self.PAD)]
        return self.sp.decode(ids)

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
