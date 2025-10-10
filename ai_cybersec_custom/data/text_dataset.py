import torch
from torch.utils.data import Dataset, DataLoader
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
import json

class TextDataset(Dataset):
    """
    FIXED: Properly handles conversational data by treating User+Agent pairs
    as single training samples with proper formatting.
    """
    def __init__(self, file_path, tokenizer, seq_len=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        print(f"📂 Loading dataset from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into conversation pairs
        conversations = content.strip().split('\n\n')  # Double newline separates conversations
        
        print(f"   Found {len(conversations)} conversation pairs")
        
        for conv in conversations:
            if not conv.strip():
                continue
                
            lines = conv.strip().split('\n')
            
            # Combine User + Agent into single sample
            if len(lines) >= 2:
                # Format: "User: question\nAgent: answer"
                full_text = '\n'.join(lines)
                
                # Encode with BOS and EOS tokens
                ids = tokenizer.encode(full_text, add_bos=True, add_eos=True)
                
                # Handle sequence length
                if len(ids) > seq_len:
                    # Truncate long sequences
                    ids = ids[:seq_len]
                else:
                    # Pad short sequences
                    ids = ids + [tokenizer.PAD] * (seq_len - len(ids))
                
                self.samples.append(torch.tensor(ids, dtype=torch.long))
        
        print(f"✅ Loaded {len(self.samples)} training samples")
        
        # Show sample
        if len(self.samples) > 0:
            sample_text = tokenizer.decode(self.samples[0].tolist())
            print(f"\n📝 Sample training example:")
            print(f"   {sample_text[:150]}...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ConversationalDataset(Dataset):
    """
    Alternative: Sliding window approach for very long conversations
    Creates overlapping windows from continuous text.
    """
    def __init__(self, file_path, tokenizer, seq_len=512, stride=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        
        print(f"📂 Loading conversational dataset from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Tokenize entire corpus
        all_tokens = tokenizer.encode(content, add_bos=True)
        
        print(f"   Total tokens: {len(all_tokens)}")
        
        # Create sliding windows
        for i in range(0, len(all_tokens) - seq_len, stride):
            window = all_tokens[i:i + seq_len]
            if len(window) == seq_len:
                self.samples.append(torch.tensor(window, dtype=torch.long))
        
        print(f"✅ Created {len(self.samples)} training windows (stride={stride})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Example usage:
if __name__ == "__main__":
    tokenizer = CustomTokenizer('../../bpe.model')
    
    # Use the fixed dataset
    dataset = TextDataset('../data/corpus.txt', tokenizer, seq_len=512)
    
    print(f"\n📊 Dataset statistics:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Sequence length: {dataset.seq_len}")
    
    # Test first sample
    sample = dataset[0]
    print(f"\n🔍 First sample shape: {sample.shape}")
    print(f"   Decoded text:\n{tokenizer.decode(sample.tolist())}")