# MINIMAL WORKING TRAINING SCRIPT
# Copy this to: ai_cybersec_custom/train/train.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import math
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.data.text_dataset import TextDataset
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, TRAIN_CONFIG, SPECIAL_TOKENS

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("❌ Installing scikit-learn...")
    os.system("pip install scikit-learn")
    from sklearn.model_selection import train_test_split


def main():
    print("="*70)
    print("🚀 MINIMAL TRANSFORMER TRAINING")
    print("="*70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(script_dir, '../tokenizer/bpe.model')
    corpus_path = os.path.join(script_dir, '../data/corpus.txt')
    checkpoint_path = TRAIN_CONFIG['checkpoint_path']
    
    # Check files exist
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print("Run: cd ai_cybersec_custom/tokenizer && python3 train_tokenizer.py")
        sys.exit(1)
    
    if not os.path.exists(corpus_path):
        print(f"❌ Corpus not found: {corpus_path}")
        sys.exit(1)
    
    # Load data
    print("📂 Loading data...")
    tokenizer = CustomTokenizer(tokenizer_path)
    dataset = TextDataset(corpus_path, tokenizer, MODEL_CONFIG['seq_len'])
    
    # Split train/val
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=TRAIN_CONFIG['batch_size'],
        sampler=torch.utils.data.SubsetRandomSampler(train_idx)
    )
    val_loader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        sampler=torch.utils.data.SubsetRandomSampler(val_idx)
    )
    
    print(f"✅ Train: {len(train_idx)}, Val: {len(val_idx)}\n")
    
    # Model
    print("🤖 Building model...")
    model = CustomTransformer(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        ff_expansion=MODEL_CONFIG['ff_expansion'],
        dropout=MODEL_CONFIG['dropout'],
        max_seq_len=MODEL_CONFIG['seq_len']
    ).to(device)
    
    print(f"✅ Parameters: {model.num_parameters:,}\n")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=TRAIN_CONFIG['lr'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['PAD'])
    
    # Create checkpoint dir
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Training loop
    print(f"🚀 Training for {TRAIN_CONFIG['epochs']} epochs...\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(TRAIN_CONFIG['epochs']):
        # Train
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = loss_fn(logits.view(-1, MODEL_CONFIG['vocab_size']), batch.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['clip'])
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % TRAIN_CONFIG['log_interval'] == 0:
                print(f"  Epoch {epoch+1}/{TRAIN_CONFIG['epochs']} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                loss = loss_fn(logits.view(-1, MODEL_CONFIG['vocab_size']), batch.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')
        
        print(f"\n📊 Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'val_ppl': val_ppl,
            }, checkpoint_path)
            print(f"✅ Saved checkpoint\n")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['patience']:
                print(f"\n⏹️  Early stopping\n")
                break
            print()
    
    print("="*70)
    print(f"✅ Training complete! Best val PPL: {math.exp(best_val_loss):.2f}")
    print(f"💾 Checkpoint: {checkpoint_path}")
    print("="*70)


if __name__ == "__main__":
    main()