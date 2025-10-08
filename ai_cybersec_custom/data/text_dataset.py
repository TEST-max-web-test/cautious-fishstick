import torch
from torch.utils.data import Dataset, DataLoader
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
import json

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128):
        self.samples = []
        self.tokenizer = tokenizer
        with open(file_path, 'r') as f:
            for line in f:
                if file_path.endswith('.jsonl'):
                    obj = json.loads(line)
                    text = obj.get('text', '')
                else:
                    text = line.strip()
                ids = tokenizer.encode(text)
                if len(ids) >= seq_len:
                    ids = ids[:seq_len]
                else:
                    ids += [0] * (seq_len - len(ids))
                self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Example usage:
# tokenizer = CustomTokenizer('tokenizer/bpe.model')
# dataset = TextDataset('data/corpus.txt', tokenizer)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
