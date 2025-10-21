#!/usr/bin/env python3
"""
COMPLETE TRAINING SCRIPT
Ready to run - no modifications needed
Place in: ai_cybersec_custom/train/train.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import math
import os
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.model.custom_transformer import ModernTransformer


class TextDataset(Dataset):
    """Load corpus and create training samples"""
    def __init__(self, file_path, tokenizer, seq_len=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        print(f"📂 Loading dataset from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into blocks (conversations or documents)
        blocks = content.strip().split('\n\n')
        print(f"   Found {len(blocks)} text blocks")
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Skip very short blocks (likely empty or just headers)
            if len(block) < 50:
                continue
            
            # Tokenize the entire block
            ids = tokenizer.encode(block, add_bos=True, add_eos=True)
            
            # Truncate or pad to seq_len
            if len(ids) > seq_len:
                ids = ids[:seq_len]
            else:
                ids = ids + [tokenizer.PAD] * (seq_len - len(ids))
            
            self.samples.append(torch.tensor(ids, dtype=torch.long))
        
        print(f"✅ Loaded {len(self.samples)} training samples\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train():
    """Main training function"""
    print("="*80)
    print("🚀 ENTERPRISE PENTESTING AI - TRAINING")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(script_dir, '../tokenizer/bpe.model')
    corpus_path = os.path.join(script_dir, '../data/combined_corpus.txt')
    checkpoint_dir = os.path.join(script_dir, 'HERE')
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    
    # Verify files exist
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print("   Run: python ai_cybersec_custom/tokenizer/train_tokenizer.py")
        sys.exit(1)
    
    if not os.path.exists(corpus_path):
        print(f"❌ Corpus not found: {corpus_path}")
        sys.exit(1)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    print("\n📂 Loading data...")
    tokenizer = CustomTokenizer(tokenizer_path)
    dataset = TextDataset(corpus_path, tokenizer, seq_len=512)
    
    # Train/val split
    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42)
    
    # Data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        pin_memory=(device.type == 'cuda'),
        num_workers=0
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        pin_memory=(device.type == 'cuda'),
        num_workers=0
    )
    
    print(f"✅ Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Build model with modern improvements
    print("\n🤖 Building model...")
    model = ModernTransformer(
        vocab_size=2000,
        hidden_size=128,  # Increased from 96
        num_layers=4,  # Increased from 3
        num_heads=4,  # Increased from 3
        num_kv_heads=2,  # GQA: 2 KV heads, 4 Q heads
        ff_expansion=4,
        dropout=0.2,  # Reduced dropout
        max_seq_len=512,
        use_gradient_checkpointing=True  # Enable for memory efficiency
    ).to(device)
    
    print(f"✅ Parameters: {model.num_parameters:,}")
    print(f"   Architecture: GQA with {model.num_kv_heads} KV heads")
    print(f"   Gradient Checkpointing: Enabled")
    
    # Optimizer with layer-wise learning rate decay
    no_decay = ['bias', 'ln', 'norm']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, betas=(0.9, 0.95))
    
    # Loss function with label smoothing
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    # Scaler for mixed precision
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Learning rate schedule: warmup + cosine decay
    warmup_epochs = 5
    total_epochs = 50
    
    def get_lr(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return 3e-4 * (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 3e-4 * 0.5 * (1 + math.cos(math.pi * progress))
    
    # Gradient accumulation for larger effective batch size
    gradient_accumulation_steps = 4
    effective_batch_size = 4 * gradient_accumulation_steps
    
    print(f"\n⚙️  Training config:")
    print(f"   Batch size: 4 (effective: {effective_batch_size} with grad accum)")
    print(f"   Epochs: {total_epochs}")
    print(f"   Learning rate: 3e-4 (warmup: {warmup_epochs} epochs)")
    print(f"   Mixed precision: {scaler is not None}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps} steps")
    
    # Training loop
    print(f"\n{'='*80}")
    print("🚀 Starting training...")
    print('='*80)
    
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    
    for epoch in range(50):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(epoch)
        
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward pass
            with autocast(enabled=(scaler is not None)):
                logits, _ = model(input_ids)
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Track
            mask = (targets != 0)
            batch_tokens = mask.sum().item()
            epoch_loss += loss.item() * batch_tokens * gradient_accumulation_steps
            epoch_tokens += batch_tokens
            
            pbar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Epoch stats
        avg_train_loss = epoch_loss / max(epoch_tokens, 1)
        train_ppl = math.exp(min(avg_train_loss, 20))
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]
                
                logits, _ = model(input_ids)
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
                
                mask = (targets != 0)
                val_tokens += mask.sum().item()
                val_loss += loss.item() * mask.sum().item()
        
        avg_val_loss = val_loss / max(val_tokens, 1)
        val_ppl = math.exp(min(avg_val_loss, 20))
        
        print(f"\n📊 Epoch {epoch+1}:")
        print(f"   Train Loss: {avg_train_loss:.4f} | PPL: {train_ppl:.2f}")
        print(f"   Val Loss:   {avg_val_loss:.4f} | PPL: {val_ppl:.2f}")
        
        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_ppl': val_ppl,
            }, checkpoint_path)
            
            print(f"   ✅ Saved checkpoint")
        else:
            no_improve += 1
            print(f"   ⏸️  No improvement ({no_improve}/{patience})")
            
            if no_improve >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
        
        # Cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"✅ Training complete!")
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    train()