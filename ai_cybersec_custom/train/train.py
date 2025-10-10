import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import math
import os

from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.data.text_dataset import TextDataset
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, TRAIN_CONFIG


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss. Caps at exp(20) for numerical stability."""
    return math.exp(loss) if loss < 20 else float('inf')


class EMA:
    """
    Exponential Moving Average (EMA):
    - Tracks a moving average of model parameters for more stable evaluation 
      and improved generalization.
    - Used during validation to apply shadow weights.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights with exponential moving average."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
    
    def apply_shadow(self):
        """Apply shadow weights to model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


# Configuration
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

# Load tokenizer and dataset
tokenizer = CustomTokenizer(os.path.join(os.path.dirname(__file__), '../../bpe.model'))
corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/corpus.txt'))
dataset = TextDataset(corpus_path, tokenizer, seq_len)

# Split into train/val
from sklearn.model_selection import train_test_split
all_indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(all_indices, test_size=0.1, random_state=42)
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# Model
model = CustomTransformer(vocab_size, hidden_size, num_layers, num_heads, ff_expansion)
model = model.cuda() if torch.cuda.is_available() else model

# Optimizer with weight decay (FIXED: added weight_decay=0.01)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# Scheduler with proper T_max calculation (FIXED: T_max is now total_steps)
total_steps = len(train_loader) * epochs
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

scaler = GradScaler()
ema = EMA(model, decay=0.999)

best_val_ppl = float('inf')

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_aux_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.cuda() if torch.cuda.is_available() else batch
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            logits, aux_loss = model(batch)
            ce_loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), batch.view(-1))
            # Combine CE loss with auxiliary load balancing loss
            loss = ce_loss + 0.01 * aux_loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA
        ema.update()
        
        total_loss += ce_loss.item()
        total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
    
    scheduler.step()
    
    # Training metrics
    train_loss = total_loss / len(train_loader)
    train_aux_loss = total_aux_loss / len(train_loader)
    train_ppl = compute_perplexity(train_loss)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
          f"Aux Loss: {train_aux_loss:.4f} | Train PPL: {train_ppl:.2f}")
    
    # Validation
    model.eval()
    ema.apply_shadow()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.cuda() if torch.cuda.is_available() else batch
            logits, aux_loss = model(batch)  # FIXED: unpacking tuple
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
        print(f"✓ Saved best checkpoint with val_ppl={val_ppl:.2f}")