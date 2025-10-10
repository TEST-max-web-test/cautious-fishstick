import torch
import torch.nn as nn
from model.custom_transformer import CustomTransformer
from tokenizer.custom_tokenizer import CustomTokenizer
from data.text_dataset import TextDataset
from torch.utils.data import DataLoader
import sys
import os
import math

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_cybersec_custom.utils.config import MODEL_CONFIG, TRAIN_CONFIG

# Configuration
vocab_size = MODEL_CONFIG['vocab_size']
hidden_size = MODEL_CONFIG['hidden_size']
num_layers = MODEL_CONFIG['num_layers']
num_heads = MODEL_CONFIG['num_heads']
ff_expansion = MODEL_CONFIG['ff_expansion']
seq_len = MODEL_CONFIG['seq_len']
batch_size = 32
checkpoint_path = TRAIN_CONFIG['checkpoint_path']

# Load tokenizer, dataset, and model
tokenizer = CustomTokenizer('../../bpe.model')
dataset = TextDataset('data/corpus.txt', tokenizer, seq_len)
loader = DataLoader(dataset, batch_size=batch_size)

model = CustomTransformer(vocab_size, hidden_size, num_layers, num_heads, ff_expansion)
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model = model.cuda() if torch.cuda.is_available() else model
model.eval()

# Evaluation metrics
loss_fn = nn.CrossEntropyLoss()
total_loss = 0
total_tokens = 0
correct = 0

with torch.no_grad():
    for batch in loader:
        batch = batch.cuda() if torch.cuda.is_available() else batch
        
        # FIXED: Unpack tuple (logits, aux_loss)
        logits, aux_loss = model(batch)
        
        loss = loss_fn(logits.view(-1, vocab_size), batch.view(-1))
        total_loss += loss.item() * batch.size(0)
        
        preds = logits.argmax(dim=-1)
        correct += (preds == batch).sum().item()
        total_tokens += batch.numel()

# Compute metrics
perplexity = math.exp(total_loss / total_tokens) if total_loss / total_tokens < 20 else float('inf')
token_accuracy = correct / total_tokens
avg_loss = total_loss / total_tokens

print(f"Perplexity: {perplexity:.4f}")
print(f"Token Accuracy: {token_accuracy:.4f}")
print(f"Average Loss: {avg_loss:.4f}")