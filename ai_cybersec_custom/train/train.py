import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import math
import os
from datetime import datetime

from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.data.text_dataset import TextDataset
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, TRAIN_CONFIG


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss. Caps at exp(20) for numerical stability."""
    return math.exp(loss) if loss < 20 else float('inf')


class WarmupScheduler:
    """Linear warmup scheduler followed by cosine annealing."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.optimizer.defaults['lr'] * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing after warmup
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.optimizer.defaults['lr'] * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


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
warmup_steps = TRAIN_CONFIG['warmup_steps']
patience = TRAIN_CONFIG['patience']
grad_accum_steps = TRAIN_CONFIG['gradient_accumulation_steps']
log_interval = TRAIN_CONFIG['log_interval']

# Device detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"📍 Device: {device}")

# Load tokenizer and dataset
print("\n📂 Loading tokenizer and dataset...")
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

print(f"✅ Loaded {len(dataset)} samples | Train: {len(train_indices)}, Val: {len(val_indices)}")

# Model
print(f"🤖 Building model: {num_layers}L x {hidden_size}D with {num_heads} heads...")
model = CustomTransformer(vocab_size, hidden_size, num_layers, num_heads, ff_expansion)
model = model.to(device)
print(f"✅ Model on {device}")

# Optimizer with weight decay
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# Scheduler with warmup
total_steps = len(train_loader) * epochs
scheduler = WarmupScheduler(optimizer, warmup_steps, total_steps)

ema = EMA(model, decay=0.999)

best_val_ppl = float('inf')
patience_counter = 0

# Training loop
print(f"\n🚀 Starting training for {epochs} epochs...\n")
start_time = datetime.now()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_aux_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (no mixed precision needed on CPU)
        logits, aux_loss = model(batch)
        ce_loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), batch.view(-1))
        loss = ce_loss + 0.01 * aux_loss
        loss = loss / grad_accum_steps  # Scale loss for accumulation
        
        # Backward pass
        loss.backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            ema.update()
        
        total_loss += ce_loss.item()
        total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            print(f"  Batch {batch_idx+1:4d}/{len(train_loader)} | Loss: {ce_loss.item():.4f} | Aux: {aux_loss.item():.4f}")
    
    # Training metrics
    train_loss = total_loss / len(train_loader)
    train_aux_loss = total_aux_loss / len(train_loader)
    train_ppl = compute_perplexity(train_loss)
    print(f"\n📊 Epoch {epoch+1}/{epochs}")
    print(f"   Train Loss: {train_loss:.4f} | Aux Loss: {train_aux_loss:.4f} | Train PPL: {train_ppl:.2f}")
    
    # Validation
    model.eval()
    ema.apply_shadow()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits, aux_loss = model(batch)
            ce_loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), batch.view(-1))
            val_loss += ce_loss.item()
    
    val_ppl = compute_perplexity(val_loss / len(val_loader))
    print(f"   Val PPL: {val_ppl:.2f}")
    
    # Save best checkpoint and early stopping
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        patience_counter = 0
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"   ✅ Saved best checkpoint (val_ppl={val_ppl:.2f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
            break

elapsed = datetime.now() - start_time
print(f"\n✅ Training complete in {elapsed}")
print(f"📊 Best validation PPL: {best_val_ppl:.2f}")
print(f"💾 Checkpoint saved to: {checkpoint_path}")