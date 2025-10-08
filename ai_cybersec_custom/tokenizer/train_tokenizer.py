
import os
from custom_tokenizer import CustomTokenizer

if __name__ == "__main__":
    # Train tokenizer on corpus.txt
    tokenizer = CustomTokenizer()
    tokenizer.train(os.path.join(os.path.dirname(__file__), '../data/corpus.txt'), 'bpe', vocab_size=100, model_type='bpe')
    print("Tokenizer trained and saved as bpe.model")
