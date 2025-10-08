import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.data.text_dataset import TextDataset
from torch.utils.data import DataLoader

import torch
from torch.utils.data import Dataset, DataLoader
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.data.text_dataset import TextDataset
import os
import json

import os

# Config
vocab_size = 100
hidden_size = 64
num_layers = 2
num_heads = 2
ff_expansion = 2
seq_len = 32
batch_size = 4
lr = 3e-4
epochs = 1
clip = 1.0
checkpoint_path = 'utils/checkpoint.pt'

from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
tokenizer = CustomTokenizer(os.path.join(os.path.dirname(__file__), '../../bpe.model'))
corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/corpus.txt'))
dataset = TextDataset(corpus_path, tokenizer, seq_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = CustomTransformer(vocab_size, hidden_size, num_layers, num_heads, ff_expansion)
model = model.cuda() if torch.cuda.is_available() else model

# Optimizer, scheduler, scaler
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
scaler = GradScaler()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.cuda() if torch.cuda.is_available() else batch
        optimizer.zero_grad()
        with autocast():
            logits = model(batch)
            loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), batch.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
    # Save checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(model.state_dict(), checkpoint_path)
