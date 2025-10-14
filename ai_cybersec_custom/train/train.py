#!/usr/bin/env python3
"""
PRODUCTION TRAINING SCRIPT
- GPU optimized with mixed precision (falls back to CPU)
- Proper autoregressive language modeling
- Scheduled sampling to combat exposure bias
- Better monitoring and checkpointing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import math
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_cybersec_custom.model.custom_transformer import ModernTransformer
from ai_cybersec_custom.data.text_dataset import TextDataset
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, TRAIN_CONFIG, SPECIAL_TOKENS

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    os.system("pip install scikit-learn")
    from sklearn.model_selection import train_test_split


def setup_device():
    """Setup device with fallback to CPU"""
    device_info = {
        'device': None,
        'use_amp': False,
        'device_name': None,
        'warning': None
    }
    
    # Try MPS first (Apple Silicon M1/M2/M3/etc)
    try:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            # Test if MPS actually works
            torch.tensor([1.0], device=device)
            device_info['device'] = device
            device_info['use_amp'] = False  # MPS doesn't support autocast reliably
            device_info['device_name'] = 'Apple Metal Performance Shaders (M1/M2/M3)'
            return device_info
    except Exception as e:
        device_info['warning'] = f"MPS available but failed to initialize: {e}"
    
    # Try CUDA (NVIDIA GPUs)
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Test if CUDA actually works
            torch.tensor([1.0], device=device)
            device_info['device'] = device
            device_info['use_amp'] = True
            device_info['device_name'] = torch.cuda.get_device_name(0)
            return device_info
    except Exception as e:
        device_info['warning'] = f"CUDA available but failed to initialize: {e}"
    
    # Fallback to CPU
    device_info['device'] = torch.device('cpu')
    device_info['use_amp'] = False
    device_info['device_name'] = 'CPU'
    if not device_info['warning']:
        device_info['warning'] = "No GPU detected - using CPU (training will be slow)"
    
    return device_info


class LabelSmoothingLoss(nn.Module):
    """Label smoothing to prevent overconfidence"""
    def __init__(self, epsilon=0.1, ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index
    
    def forward(self, logits, target):
        n_classes = logits.size(-1)
        
        # One-hot encode target
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.epsilon / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)
            
            # Mask padding
            mask = target != self.ignore_index
            true_dist = true_dist * mask.unsqueeze(1)
        
        # Cross entropy with smooth labels
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(true_dist * log_probs).sum(dim=-1)
        
        return loss[mask].mean()


def get_lr_schedule(optimizer, warmup_steps, total_steps):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for batch in val_loader:
        batch = batch.to(device)
        
        # Shift for autoregressive LM
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Forward
        logits, _ = model(input_ids)
        
        # Loss
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Track
        mask = (targets != SPECIAL_TOKENS['PAD'])
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # Cap for numerical stability
    
    return avg_loss, perplexity


@torch.no_grad()
def test_generation(model, tokenizer, device, prompts):
    """Test generation quality"""
    model.eval()
    results = []
    
    for prompt in prompts:
        formatted = f"User: {prompt}\nAgent:"
        input_ids = torch.tensor([tokenizer.encode(formatted, add_bos=True)], device=device)
        
        # Generate
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        # Decode response only
        response_ids = output[0, input_ids.size(1):].tolist()
        response = tokenizer.decode(response_ids).strip()
        
        results.append((prompt, response))
    
    return results


def main():
    print("="*80)
    print("🚀 PRODUCTION TRAINING - GPU OPTIMIZED WITH CPU FALLBACK")
    print("="*80)
    
    # Setup device with fallback
    device_info = setup_device()
    device = device_info['device']
    use_amp = device_info['use_amp']
    
    print(f"\n📍 Device: {device}")
    print(f"   Type: {device_info['device_name']}")
    
    if device_info['warning']:
        print(f"   ⚠️  {device_info['warning']}")
    
    if device.type == 'cuda':
        try:
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except:
            pass
    
    print(f"   Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(script_dir, '../tokenizer/bpe.model')
    corpus_path = os.path.join(script_dir, '../data/corpus.txt')
    checkpoint_path = TRAIN_CONFIG['checkpoint_path']
    
    # Check files
    if not os.path.exists(tokenizer_path):
        print(f"\n❌ Tokenizer not found: {tokenizer_path}")
        sys.exit(1)
    if not os.path.exists(corpus_path):
        print(f"❌ Corpus not found: {corpus_path}")
        sys.exit(1)
    
    # Load data
    print("\n📂 Loading data...")
    tokenizer = CustomTokenizer(tokenizer_path)
    dataset = TextDataset(corpus_path, tokenizer, MODEL_CONFIG['seq_len'])
    
    # Split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42)
    
    pin_memory = (device.type == 'cuda')
    
    train_loader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        pin_memory=pin_memory,
        num_workers=0  # Avoid multiprocessing issues
    )
    val_loader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        pin_memory=pin_memory
    )
    
    print(f"✅ Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    
    # Model
    print("\n🤖 Building model...")
    model = ModernTransformer(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        ff_expansion=MODEL_CONFIG['ff_expansion'],
        dropout=MODEL_CONFIG['dropout'],
        max_seq_len=MODEL_CONFIG['seq_len']
    ).to(device)
    
    print(f"✅ Parameters: {model.num_parameters:,}")
    print(f"   Memory: ~{model.num_parameters * 4 / 1e6:.1f} MB (FP32)")
    
    # Optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['lr'],
        betas=(0.9, 0.95),  # GPT-3 uses 0.95 for beta2
        weight_decay=TRAIN_CONFIG['weight_decay'],
        eps=1e-8
    )
    
    # Learning rate schedule
    total_steps = len(train_loader) * TRAIN_CONFIG['epochs']
    scheduler = get_lr_schedule(optimizer, TRAIN_CONFIG['warmup_steps'], total_steps)
    
    # Loss with label smoothing
    loss_fn = LabelSmoothingLoss(
        epsilon=TRAIN_CONFIG.get('label_smoothing', 0.1),
        ignore_index=SPECIAL_TOKENS['PAD']
    )
    
    # Mixed precision training (GPU only)
    scaler = GradScaler() if use_amp else None
    
    print(f"\n⚙️  Training config:")
    print(f"   Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"   Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"   Learning rate: {TRAIN_CONFIG['lr']}")
    print(f"   Warmup steps: {TRAIN_CONFIG['warmup_steps']}")
    print(f"   Mixed precision: {use_amp}")
    print(f"   Label smoothing: {TRAIN_CONFIG.get('label_smoothing', 0.1)}")
    
    # Create checkpoint directory
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Training loop
    print(f"\n{'='*80}")
    print("🚀 Starting training...")
    print('='*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    for epoch in range(TRAIN_CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        epoch_tokens = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            
            # Autoregressive LM: predict next token
            # Input: [BOS, tok1, tok2, tok3, ...]
            # Target: [tok1, tok2, tok3, ..., EOS]
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward pass with mixed precision
            with autocast(enabled=use_amp):
                logits, _ = model(input_ids)
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
            
            # Backward pass
            optimizer.zero_grad()
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['clip'])
                optimizer.step()
            
            scheduler.step()
            
            # Track metrics
            mask = (targets != SPECIAL_TOKENS['PAD'])
            batch_tokens = mask.sum().item()
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Epoch metrics
        avg_train_loss = epoch_loss / epoch_tokens
        train_ppl = math.exp(min(avg_train_loss, 20))
        
        # Validation
        val_loss, val_ppl = evaluate(model, val_loader, loss_fn, device)
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        
        # Test generation every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n🧪 Testing generation:")
            test_prompts = ["Hello", "What is SQL injection?", "What is XSS?"]
            results = test_generation(model, tokenizer, device, test_prompts)
            
            for prompt, response in results:
                print(f"   Q: {prompt}")
                print(f"   A: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'config': MODEL_CONFIG
            }, checkpoint_path)
            
            print(f"   ✅ Saved checkpoint (Val PPL: {val_ppl:.2f})")
        else:
            patience_counter += 1
            print(f"   ⏸️  No improvement ({patience_counter}/{TRAIN_CONFIG['patience']})")
            
            if patience_counter >= TRAIN_CONFIG['patience']:
                print(f"\n⏹️  Early stopping triggered")
                break
        
        print()
    
    # Final evaluation
    print("="*80)
    print("🎯 FINAL EVALUATION")
    print("="*80)
    
    # Load best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nBest checkpoint:")
    print(f"   Epoch: {checkpoint['epoch']+1}")
    print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"   Val PPL: {checkpoint['val_ppl']:.2f}")
    
    # Comprehensive generation test
    print(f"\n🎤 Generation samples:")
    test_prompts = [
        "Hello",
        "What is SQL injection?",
        "What is XSS?",
        "How do I stay ethical?",
        "What is pentesting?",
        "Tell me about cybersecurity"
    ]
    
    results = test_generation(model, tokenizer, device, test_prompts)
    
    for i, (prompt, response) in enumerate(results, 1):
        print(f"\n{i}. Q: {prompt}")
        print(f"   A: {response}")
    
    print(f"\n{'='*80}")
    print(f"✅ Training complete!")
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    if checkpoint['val_ppl'] > 10:
        print("⚠️  High perplexity - model may need:")
        print("   1. More training epochs")
        print("   2. More/better training data")
        print("   3. Different hyperparameters")
    elif checkpoint['val_ppl'] < 3:
        print("✅ Low perplexity - model learned well!")
        if all(len(r[1]) < 10 for r in results[:3]):
            print("   ⚠️  But responses are short - may need:")
            print("      - Lower repetition penalty")
            print("      - Different sampling parameters")


if __name__ == "__main__":
    main()