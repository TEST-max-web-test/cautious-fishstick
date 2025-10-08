import torch
import torch.nn as nn
from model.custom_transformer import CustomTransformer
from tokenizer.custom_tokenizer import CustomTokenizer
from data.text_dataset import TextDataset
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from model.custom_transformer import CustomTransformer
from tokenizer.custom_tokenizer import CustomTokenizer
from data.text_dataset import TextDataset
from torch.utils.data import DataLoader
import math

# Config
vocab_size = 100
hidden_size = 512
num_layers = 6
num_heads = 8
ff_expansion = 4
seq_len = 128
batch_size = 32
checkpoint_path = 'utils/checkpoint.pt'

tokenizer = CustomTokenizer('../../bpe.model')
dataset = TextDataset('data/corpus.txt', tokenizer, seq_len)
loader = DataLoader(dataset, batch_size=batch_size)

model = CustomTransformer(vocab_size, hidden_size, num_layers, num_heads, ff_expansion)
model.load_state_dict(torch.load(checkpoint_path))
model = model.cuda() if torch.cuda.is_available() else model
model.eval()

# Metrics
loss_fn = nn.CrossEntropyLoss()
total_loss = 0
total_tokens = 0
correct = 0

with torch.no_grad():
    for batch in loader:
        batch = batch.cuda() if torch.cuda.is_available() else batch
        logits = model(batch)
        loss = loss_fn(logits.view(-1, vocab_size), batch.view(-1))
        total_loss += loss.item() * batch.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch).sum().item()
        total_tokens += batch.numel()

perplexity = math.exp(total_loss / total_tokens)
token_accuracy = correct / total_tokens
avg_loss = total_loss / total_tokens

print(f"Perplexity: {perplexity:.4f}")
print(f"Token Accuracy: {token_accuracy:.4f}")
print(f"Average Loss: {avg_loss:.4f}")
