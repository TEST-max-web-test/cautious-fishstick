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

import math
import json

import os


# Config
from ai_cybersec_custom.utils.config import MODEL_CONFIG, TRAIN_CONFIG
vocab_size = MODEL_CONFIG['vocab_size']
hidden_size = MODEL_CONFIG['hidden_size']
num_layers = MODEL_CONFIG['num_layers']
num_heads = MODEL_CONFIG['num_heads']
ff_expansion = MODEL_CONFIG['ff_expansion']
seq_len = MODEL_CONFIG['seq_len']
batch_size = TRAIN_CONFIG['batch_size']
lr = TRAIN_CONFIG['lr']
epochs = TRAIN_CONFIG['epochs']
clip = TRAIN_CONFIG['clip']
checkpoint_path = TRAIN_CONFIG['checkpoint_path']

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


# Validation split
from sklearn.model_selection import train_test_split
all_indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(all_indices, test_size=0.1, random_state=42)
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# EMA
class EMA:
    """
    Exponential Moving Average (EMA):
    - Tracks a moving average of model parameters for more stable evaluation and improved generalization.
    - Used during validation to apply shadow weights.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

ema = EMA(model)

def compute_perplexity(loss):
    """
    Computes perplexity from loss value. Caps at exp(20) for stability.
    """
    return math.exp(loss) if loss < 20 else float('inf')

best_val_ppl = float('inf')
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_aux_loss = 0
    for batch in train_loader:
        batch = batch.cuda() if torch.cuda.is_available() else batch
        optimizer.zero_grad()
        with autocast():
            logits, aux_loss = model(batch)
            ce_loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), batch.view(-1))
            loss = ce_loss + 0.01 * aux_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update()
        total_loss += ce_loss.item()
        total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
    scheduler.step()
    train_ppl = compute_perplexity(total_loss / len(train_loader))
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Aux Loss: {total_aux_loss/len(train_loader):.4f} | Train PPL: {train_ppl:.2f}")

    # Validation
    model.eval()
    ema.apply_shadow()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.cuda() if torch.cuda.is_available() else batch
            logits, aux_loss = model(batch)
            ce_loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), batch.view(-1))
            val_loss += ce_loss.item()
    val_ppl = compute_perplexity(val_loss / len(val_loader))
    print(f"Epoch {epoch+1}/{epochs} | Val PPL: {val_ppl:.2f}")
    # Save best checkpoint
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(model.state_dict(), checkpoint_path)
