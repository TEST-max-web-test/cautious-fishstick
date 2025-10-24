#!/usr/bin/env python3
"""
TRAINING SCRIPT FOR MOE TRANSFORMER
Train the Mixture of Experts model on cybersecurity data
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
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.model.moe_transformer import MoETransformer


class TextDataset(Dataset):
    """Load corpus and create training samples"""
    def __init__(self, file_path, tokenizer, seq_len=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        print(f"📂 Loading dataset from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into blocks
        blocks = content.strip().split('\n\n')
        print(f"   Found {len(blocks)} text blocks")
        
        for block in blocks:
            block = block.strip()
            if len(block) < 50:
                continue
            
            # Tokenize
            ids = tokenizer.encode(block, add_bos=True, add_eos=True)
            
            # Create samples with sliding window
            for i in range(0, len(ids), seq_len // 2):  # 50% overlap
                sample = ids[i:i + seq_len]
                
                if len(sample) < seq_len // 2:  # Skip very short samples
                    continue
                
                # Pad if needed
                if len(sample) < seq_len:
                    sample = sample + [tokenizer.PAD] * (seq_len - len(sample))
                
                self.samples.append(torch.tensor(sample, dtype=torch.long))
        
        print(f"✅ Loaded {len(self.samples)} training samples\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train():
    """Main training function for MoE"""
    print("="*80)
    print("🚀 MIXTURE OF EXPERTS TRANSFORMER - TRAINING")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Paths
    script_dir = Path(__file__).parent
    tokenizer_path = script_dir / '../tokenizer/bpe.model'
    corpus_path = script_dir / '../data/combined_corpus.txt'
    checkpoint_dir = script_dir / 'checkpoints'
    checkpoint_path = checkpoint_dir / 'moe_checkpoint.pt'
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Verify files
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print("   Run: python ai_cybersec_custom/tokenizer/train_tokenizer.py")
        sys.exit(1)
    
    if not corpus_path.exists():
        print(f"❌ Corpus not found: {corpus_path}")
        print("   Make sure to run the scraper and filter data first")
        sys.exit(1)
    
    # Load data
    print("\n📂 Loading data...")
    tokenizer = CustomTokenizer(str(tokenizer_path))
    dataset = TextDataset(str(corpus_path), tokenizer, seq_len=512)
    
    # Train/val split
    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    # Data loaders
    batch_size = 2 if device.type == 'cuda' else 1  # Small batch for MoE
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        pin_memory=(device.type == 'cuda'),
        num_workers=0
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        pin_memory=(device.type == 'cuda'),
        num_workers=0
    )
    
    print(f"✅ Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Build MoE model
    print("\n🤖 Building Mixture of Experts model...")
    model = MoETransformer(
        vocab_size=2000,
        hidden_size=256,
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,  # GQA
        num_experts=8,
        top_k_experts=2,
        ff_expansion=4,
        dropout=0.1,
        max_seq_len=512,
        use_gradient_checkpointing=True,
        use_flash_attention=True,
        aux_loss_weight=0.01
    ).to(device)
    
    print(f"✅ Parameters: {model.num_parameters:,}")
    print(f"   Architecture: 8 experts, top-2 routing, GQA")
    print(f"   Flash Attention: Enabled")
    print(f"   Gradient Checkpointing: Enabled")
    
    # Optimizer
    no_decay = ['bias', 'norm', 'ln']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n.lower() for nd in no_decay)],
            'weight_decay': 0.01,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n.lower() for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-4, betas=(0.9, 0.95))
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    # Mixed precision
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Training config
    warmup_epochs = 3
    total_epochs = 40
    gradient_accumulation_steps = 8
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return 2e-4 * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 2e-4 * 0.5 * (1 + math.cos(math.pi * progress))
    
    print(f"\n⚙️  Training config:")
    print(f"   Batch size: {batch_size} (effective: {effective_batch_size})")
    print(f"   Epochs: {total_epochs}")
    print(f"   Learning rate: 2e-4 (warmup: {warmup_epochs})")
    print(f"   Mixed precision: {scaler is not None}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    
    # Training loop
    print(f"\n{'='*80}")
    print("🚀 Starting training...")
    print('='*80)
    
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    
    for epoch in range(total_epochs):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(epoch)
        
        model.train()
        epoch_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_tokens = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward pass
            with autocast(enabled=(scaler is not None)):
                logits, aux_loss = model(input_ids, return_aux_loss=True)
                
                # Main loss
                main_loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
                
                # Total loss (main + auxiliary load balancing)
                loss = main_loss + aux_loss
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
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
            epoch_loss += main_loss.item() * batch_tokens * gradient_accumulation_steps
            epoch_aux_loss += aux_loss.item() * gradient_accumulation_steps
            epoch_tokens += batch_tokens
            
            pbar.set_postfix({
                'loss': f"{main_loss.item() * gradient_accumulation_steps:.4f}",
                'aux': f"{aux_loss.item() * gradient_accumulation_steps:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Epoch stats
        avg_train_loss = epoch_loss / max(epoch_tokens, 1)
        avg_aux_loss = epoch_aux_loss / max(len(train_loader), 1)
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
                
                logits, _ = model(input_ids, return_aux_loss=False)
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
        print(f"   Aux Loss:   {avg_aux_loss:.6f}")
        
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
                'config': {
                    'vocab_size': 2000,
                    'hidden_size': 256,
                    'num_layers': 8,
                    'num_heads': 8,
                    'num_kv_heads': 2,
                    'num_experts': 8,
                    'top_k_experts': 2,
                }
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
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    train()
