#!/usr/bin/env python3
"""
IMPROVED TRAINING SCRIPT
Addresses overfitting and generation issues with:
- Label smoothing to prevent overconfidence
- Higher dropout during training
- Validation on unseen prompts
- Better monitoring of generation quality
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.data.text_dataset import TextDataset
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, TRAIN_CONFIG, SPECIAL_TOKENS

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    os.system("pip install scikit-learn")
    from sklearn.model_selection import train_test_split


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing to prevent overconfidence
    Instead of 100% probability on correct token, distribute some to others
    """
    def __init__(self, epsilon=0.1, ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index
    
    def forward(self, logits, target):
        # logits: [batch * seq_len, vocab_size]
        # target: [batch * seq_len]
        
        n_classes = logits.size(-1)
        
        # Create smoothed labels
        with torch.no_grad():
            # Start with uniform distribution
            smooth_target = torch.full_like(logits, self.epsilon / (n_classes - 1))
            
            # Reshape target for scatter
            target_reshaped = target.unsqueeze(1)
            
            # Put (1 - epsilon) probability on correct class
            smooth_target.scatter_(1, target_reshaped, 1.0 - self.epsilon)
            
            # Mask out ignored indices (like PAD tokens)
            mask = (target != self.ignore_index).unsqueeze(1)
            smooth_target = smooth_target * mask
        
        # Compute cross entropy with smoothed labels
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_target * log_probs).sum(dim=-1)
        
        # Mask ignored indices
        mask = (target != self.ignore_index)
        loss = loss * mask
        
        return loss.sum() / mask.sum()


def test_generation(model, tokenizer, device, prompts=None):
    """
    Test generation quality during training
    """
    if prompts is None:
        prompts = [
            "What is SQL injection?",
            "Hello",
            "What is XSS?",
        ]
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            formatted = f"User: {prompt}\nAgent:"
            ids = tokenizer.encode(formatted, add_bos=True)
            prompt_length = len(ids)
            
            # Generate up to 30 tokens
            for _ in range(30):
                x = torch.tensor([ids], device=device)
                if x.size(1) > MODEL_CONFIG['seq_len']:
                    x = x[:, -MODEL_CONFIG['seq_len']:]
                
                logits, _ = model(x)
                next_token = logits[0, -1].argmax().item()
                
                if next_token in (tokenizer.EOS, tokenizer.PAD):
                    break
                
                ids.append(next_token)
            
            response_ids = ids[prompt_length:]
            response = tokenizer.decode(response_ids).strip()
            results.append((prompt, response))
    
    return results


def main():
    print("="*70)
    print("🚀 IMPROVED TRANSFORMER TRAINING")
    print("="*70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(script_dir, '../tokenizer/bpe.model')
    corpus_path = os.path.join(script_dir, '../data/corpus.txt')
    checkpoint_path = TRAIN_CONFIG['checkpoint_path']
    
    # Check files
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        sys.exit(1)
    if not os.path.exists(corpus_path):
        print(f"❌ Corpus not found: {corpus_path}")
        sys.exit(1)
    
    # Load data
    print("📂 Loading data...")
    tokenizer = CustomTokenizer(tokenizer_path)
    dataset = TextDataset(corpus_path, tokenizer, MODEL_CONFIG['seq_len'])
    
    # Split train/val (larger val set for better estimation)
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=TRAIN_CONFIG['batch_size'],
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        drop_last=True  # Drop incomplete batches
    )
    val_loader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        sampler=torch.utils.data.SubsetRandomSampler(val_idx)
    )
    
    print(f"✅ Train: {len(train_idx)}, Val: {len(val_idx)}\n")
    
    # Model with INCREASED dropout for regularization
    print("🤖 Building model...")
    model = CustomTransformer(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        ff_expansion=MODEL_CONFIG['ff_expansion'],
        dropout=0.2,  # Increased from 0.1 to combat overfitting
        max_seq_len=MODEL_CONFIG['seq_len']
    ).to(device)
    
    print(f"✅ Parameters: {model.num_parameters:,}\n")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=TRAIN_CONFIG['lr'],
        weight_decay=0.05  # Increased weight decay
    )
    
    # Loss with label smoothing
    label_smoothing = TRAIN_CONFIG.get('label_smoothing', 0.1)
    print(f"Using label smoothing: {label_smoothing}")
    loss_fn = LabelSmoothingCrossEntropy(
        epsilon=label_smoothing,
        ignore_index=SPECIAL_TOKENS['PAD']
    )
    
    # Create checkpoint dir
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Training loop
    print(f"\n🚀 Training for {TRAIN_CONFIG['epochs']} epochs...\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(TRAIN_CONFIG['epochs']):
        # Train
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch)
            
            # Reshape for loss
            logits_flat = logits.view(-1, MODEL_CONFIG['vocab_size'])
            target_flat = batch.view(-1)
            
            loss = loss_fn(logits_flat, target_flat)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['clip'])
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if (batch_idx + 1) % TRAIN_CONFIG['log_interval'] == 0:
                print(f"  Epoch {epoch+1}/{TRAIN_CONFIG['epochs']} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                
                logits_flat = logits.view(-1, MODEL_CONFIG['vocab_size'])
                target_flat = batch.view(-1)
                
                loss = loss_fn(logits_flat, target_flat)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')
        
        print(f"\n📊 Epoch {epoch+1}:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {avg_val_loss:.4f}")
        print(f"   Val PPL:    {val_ppl:.2f}")
        
        # Test generation every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n🧪 Testing generation:")
            results = test_generation(model, tokenizer, device)
            for prompt, response in results:
                print(f"   Q: {prompt}")
                print(f"   A: {response[:80]}{'...' if len(response) > 80 else ''}")
            print()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_ppl': val_ppl,
            }, checkpoint_path)
            print(f"✅ Saved checkpoint (Val PPL: {val_ppl:.2f})\n")
        else:
            patience_counter += 1
            print(f"⏸️  No improvement ({patience_counter}/{TRAIN_CONFIG['patience']})\n")
            
            if patience_counter >= TRAIN_CONFIG['patience']:
                print(f"⏹️  Early stopping\n")
                break
    
    # Final generation test
    print("="*70)
    print("🎯 FINAL GENERATION TEST")
    print("="*70)
    
    # Load best checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    test_prompts = [
        "What is SQL injection?",
        "Hello",
        "What is XSS?",
        "How do I stay ethical?",
        "What is pentesting?",
    ]
    
    results = test_generation(model, tokenizer, device, test_prompts)
    for prompt, response in results:
        print(f"\nQ: {prompt}")
        print(f"A: {response}")
    
    print("\n" + "="*70)
    print(f"✅ Training complete! Best val PPL: {math.exp(best_val_loss):.2f}")
    print(f"💾 Checkpoint: {checkpoint_path}")
    print("="*70)
    print("\n⚠️  If responses are still bad, the model may need:")
    print("   1. More diverse training data")
    print("   2. Different architecture (larger model)")
    print("   3. Different training approach (curriculum learning)")


if __name__ == "__main__":
    main()