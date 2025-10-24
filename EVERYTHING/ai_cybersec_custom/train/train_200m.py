#!/usr/bin/env python3
"""
TRAINING SCRIPT FOR 200M PARAMETER MOE TRANSFORMER
Train the large-scale Mixture of Experts model on comprehensive cybersecurity data

This script is optimized for:
- 200M parameter model (50M active per token)
- Large-scale data (500MB-1GB+ corpus)
- Multi-GPU training (if available)
- Memory-efficient training with gradient checkpointing
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
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.model.moe_transformer_200m import MoETransformer200M


class TextDataset(Dataset):
    """Load corpus and create training samples with sliding window"""
    def __init__(self, file_path, tokenizer, seq_len=2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        print(f"📂 Loading dataset from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into blocks
        blocks = content.strip().split('\n\n')
        print(f"   Found {len(blocks)} text blocks")
        
        # Process blocks with sliding window for maximum data utilization
        for block in tqdm(blocks, desc="Processing blocks"):
            block = block.strip()
            if len(block) < 100:  # Skip very short blocks
                continue
            
            # Tokenize
            ids = tokenizer.encode(block, add_bos=True, add_eos=True)
            
            # Create overlapping samples for better learning
            stride = seq_len // 2  # 50% overlap
            for i in range(0, len(ids), stride):
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
    """Main training function for 200M parameter MoE"""
    print("="*80)
    print("🚀 200M PARAMETER MIXTURE OF EXPERTS - TRAINING")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Paths
    script_dir = Path(__file__).parent
    tokenizer_path = script_dir / '../tokenizer/bpe.model'
    corpus_path = script_dir / '../data/combined_corpus.txt'
    checkpoint_dir = script_dir / 'checkpoints'
    checkpoint_path = checkpoint_dir / 'moe_200m_checkpoint.pt'
    best_checkpoint_path = checkpoint_dir / 'moe_200m_best.pt'
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Verify files
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print("   Creating tokenizer with larger vocabulary for 200M model...")
        # Create tokenizer with 32k vocab
        sys.exit(1)
    
    if not corpus_path.exists():
        print(f"❌ Corpus not found: {corpus_path}")
        print("   Please wait for the comprehensive scraper to finish")
        print("   Or use existing corpus")
        sys.exit(1)
    
    # Check corpus size
    corpus_size = corpus_path.stat().st_size / (1024 * 1024)
    print(f"\n📊 Corpus size: {corpus_size:.2f} MB")
    
    if corpus_size < 100:
        print("⚠️  Warning: Corpus is smaller than recommended for 200M model")
        print("   Recommended: 500MB-1GB+ for optimal training")
        print("   Current corpus will work but may underutilize model capacity")
    
    # Load data
    print("\n📂 Loading data...")
    tokenizer = CustomTokenizer(str(tokenizer_path))
    print(f"   Tokenizer vocabulary: {tokenizer.vocab_size}")
    
    # Use longer sequences for 200M model
    dataset = TextDataset(str(corpus_path), tokenizer, seq_len=2048)
    
    if len(dataset) < 1000:
        print("⚠️  Warning: Small dataset detected")
        print(f"   Only {len(dataset)} samples available")
        print("   200M model works best with 50k+ samples")
    
    # Train/val split
    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.05, random_state=42)
    
    # Data loaders - very small batch for 200M model
    batch_size = 1  # Must be 1 for 200M model
    
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
    
    # Build 200M MoE model
    print("\n🤖 Building 200M Parameter Mixture of Experts model...")
    print("   This may take a minute...")
    
    model = MoETransformer200M(
        vocab_size=tokenizer.vocab_size,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        num_kv_heads=4,     # GQA: 4:1 ratio
        num_experts=32,     # 32 experts
        top_k_experts=4,    # Top-4 routing
        ff_expansion=4,
        dropout=0.1,
        max_seq_len=2048,
        use_gradient_checkpointing=True,
        use_flash_attention=True,
        aux_loss_weight=0.01
    ).to(device)
    
    print(f"✅ Model created!")
    print(f"   Total Parameters: {model.num_parameters:,}")
    print(f"   Active per token: ~{model.num_parameters // 4:,} (25%)")
    print(f"   Architecture: 32 experts, top-4 routing, GQA 4:1")
    print(f"   Context length: 2048 tokens")
    print(f"   Flash Attention: Enabled")
    print(f"   Gradient Checkpointing: Enabled")
    
    # Optimizer with proper weight decay
    no_decay = ['bias', 'norm', 'ln']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n.lower() for nd in no_decay)],
            'weight_decay': 0.1,  # Higher weight decay for large model
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n.lower() for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    # Mixed precision
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Training config
    warmup_epochs = 5
    total_epochs = 30
    gradient_accumulation_steps = 32  # Large accumulation for stability
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return 1e-4 * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 1e-4 * 0.1 * (1 + math.cos(math.pi * progress))  # Decay to 10%
    
    print(f"\n⚙️  Training configuration:")
    print(f"   Batch size: {batch_size} (effective: {effective_batch_size})")
    print(f"   Epochs: {total_epochs}")
    print(f"   Learning rate: 1e-4 (warmup: {warmup_epochs})")
    print(f"   Mixed precision: {scaler is not None}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps} steps")
    print(f"   Weight decay: 0.1")
    print(f"   Label smoothing: 0.1")
    
    # Estimate training time
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * total_epochs
    print(f"\n📊 Training estimates:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Estimated time: 12-24 hours (depends on GPU)")
    
    # Training loop
    print(f"\n{'='*80}")
    print("🚀 Starting training...")
    print('='*80)
    
    best_val_loss = float('inf')
    patience = 15
    no_improve = 0
    training_stats = []
    
    for epoch in range(total_epochs):
        # Update learning rate
        current_lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
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
                
                main_loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
                
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
                'lr': f"{current_lr:.2e}"
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
            for batch in tqdm(val_loader, desc="Validation"):
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
        
        # Log
        stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_ppl': train_ppl,
            'val_loss': avg_val_loss,
            'val_ppl': val_ppl,
            'aux_loss': avg_aux_loss,
            'lr': current_lr
        }
        training_stats.append(stats)
        
        print(f"\n📊 Epoch {epoch+1}:")
        print(f"   Train Loss: {avg_train_loss:.4f} | PPL: {train_ppl:.2f}")
        print(f"   Val Loss:   {avg_val_loss:.4f} | PPL: {val_ppl:.2f}")
        print(f"   Aux Loss:   {avg_aux_loss:.6f}")
        print(f"   LR:         {current_lr:.2e}")
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_ppl': val_ppl,
            'stats': training_stats,
            'config': {
                'vocab_size': tokenizer.vocab_size,
                'hidden_size': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'num_kv_heads': 4,
                'num_experts': 32,
                'top_k_experts': 4,
            }
        }, checkpoint_path)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_ppl': val_ppl,
                'stats': training_stats,
                'config': {
                    'vocab_size': tokenizer.vocab_size,
                    'hidden_size': 1024,
                    'num_layers': 24,
                    'num_heads': 16,
                    'num_kv_heads': 4,
                    'num_experts': 32,
                    'top_k_experts': 4,
                }
            }, best_checkpoint_path)
            
            print(f"   ✅ Saved best checkpoint (val_loss: {avg_val_loss:.4f})")
        else:
            no_improve += 1
            print(f"   ⏸️  No improvement ({no_improve}/{patience})")
            
            if no_improve >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
        
        # Save training stats
        with open(checkpoint_dir / 'training_stats.json', 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        # Cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"✅ Training complete!")
    print(f"💾 Best checkpoint: {best_checkpoint_path}")
    print(f"💾 Latest checkpoint: {checkpoint_path}")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"📈 Training stats: {checkpoint_dir / 'training_stats.json'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    train()
