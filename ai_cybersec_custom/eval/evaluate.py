import torch
import torch.nn as nn
from model.custom_transformer import CustomTransformer
from tokenizer.custom_tokenizer import CustomTokenizer
from data.text_dataset import TextDataset
from torch.utils.data import DataLoader
import sys
import os
import math

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_cybersec_custom.utils.config import MODEL_CONFIG, TRAIN_CONFIG


def evaluate_model(checkpoint_path: str = None):
    """
    Evaluate model on test set.
    
    Metrics computed:
    - Perplexity: Lower is better
    - Token Accuracy: % of correctly predicted tokens
    - Cross-Entropy Loss: Average loss
    """
    
    # Configuration
    vocab_size = MODEL_CONFIG['vocab_size']
    hidden_size = MODEL_CONFIG['hidden_size']
    num_layers = MODEL_CONFIG['num_layers']
    num_heads = MODEL_CONFIG['num_heads']
    ff_expansion = MODEL_CONFIG['ff_expansion']
    seq_len = MODEL_CONFIG['seq_len']
    batch_size = 32
    
    checkpoint_path = checkpoint_path or TRAIN_CONFIG['checkpoint_path']
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📍 Device: {device}\n")
    
    # Load tokenizer, dataset
    print("📂 Loading data...")
    tokenizer = CustomTokenizer('../../bpe.model')
    dataset = TextDataset('data/corpus.txt', tokenizer, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size)
    print(f"✅ Loaded {len(dataset)} samples\n")
    
    # Load model
    print(f"🤖 Loading model from {checkpoint_path}...")
    model = CustomTransformer(vocab_size, hidden_size, num_layers, num_heads, ff_expansion)
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Model loaded\n")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("Using untrained model\n")
    
    model = model.to(device)
    model.eval()
    
    # Evaluation loop
    print("🔍 Evaluating...\n")
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            
            # Forward pass
            logits, aux_loss = model(batch)
            loss = loss_fn(logits.view(-1, vocab_size), batch.view(-1))
            
            # Metrics
            total_loss += loss.item() * batch.size(0)
            preds = logits.argmax(dim=-1)
            correct = (preds == batch).sum().item()
            total_correct += correct
            total_tokens += batch.numel()
            batch_count += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1:3d}/{len(loader)} | Loss: {loss.item():.4f}")
    
    # Compute metrics
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    token_accuracy = total_correct / total_tokens
    
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Average Loss:      {avg_loss:.4f}")
    print(f"Perplexity:        {perplexity:.2f}")
    print(f"Token Accuracy:    {token_accuracy:.4f} ({total_correct:,}/{total_tokens:,})")
    print(f"Model Parameters:  {model.num_parameters:,}")
    print("="*60)
    
    # Interpretation
    if perplexity < 5:
        print("✅ Excellent! Model is learning well.")
    elif perplexity < 10:
        print("👍 Good! Model is converging.")
    elif perplexity < 50:
        print("🤔 Reasonable. More training might help.")
    else:
        print("⚠️  Model needs more training.")
    
    return {
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'token_accuracy': token_accuracy,
        'total_tokens': total_tokens
    }


if __name__ == "__main__":
    checkpoint_path = TRAIN_CONFIG['checkpoint_path']
    evaluate_model(checkpoint_path)