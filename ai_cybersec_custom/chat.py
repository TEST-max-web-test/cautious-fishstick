# MINIMAL WORKING CHAT SCRIPT
# Copy this to: ai_cybersec_custom/chat.py

import torch
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, INFER_CONFIG


def load_model(device='cpu'):
    """Load model and tokenizer"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find tokenizer
    tokenizer_path = os.path.join(script_dir, 'tokenizer/bpe.model')
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        sys.exit(1)
    
    tokenizer = CustomTokenizer(tokenizer_path)
    
    # Create model
    model = CustomTransformer(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        ff_expansion=MODEL_CONFIG['ff_expansion'],
        dropout=MODEL_CONFIG['dropout'],
        max_seq_len=MODEL_CONFIG['seq_len']
    ).to(device)
    
    # Load checkpoint
    checkpoint_paths = [
        'ai_cybersec_custom/train/utils/checkpoint.pt',
        os.path.join(script_dir, 'train/utils/checkpoint.pt'),
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"✅ Loaded checkpoint")
    else:
        print("⚠️  No checkpoint found - using untrained model")
    
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, device='cpu'):
    """Generate response"""
    max_len = INFER_CONFIG['max_response_length']
    temp = INFER_CONFIG['temperature']
    top_k = INFER_CONFIG['top_k']
    
    # Encode
    formatted = f"User: {prompt}\nAgent:"
    ids = tokenizer.encode(formatted, add_bos=True)
    
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([ids], device=device)
            
            # Truncate if too long
            if x.size(1) > MODEL_CONFIG['seq_len']:
                x = x[:, -MODEL_CONFIG['seq_len']:]
            
            # Forward
            logits, _ = model(x)
            logits = logits[0, -1] / temp
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, len(logits)))
                logits[logits < v[-1]] = -float('inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop conditions
            if next_token in (tokenizer.EOS, tokenizer.PAD):
                break
            
            ids.append(next_token)
    
    # Decode
    response_ids = ids[len(tokenizer.encode(formatted, add_bos=True)):]
    response = tokenizer.decode(response_ids).strip()
    
    # Clean up
    if "User:" in response:
        response = response.split("User:")[0].strip()
    
    return response


def main():
    print("="*70)
    print("💬 CHAT INTERFACE")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("Commands: 'exit' to quit, 'clear' to reset")
    print("="*70 + "\n")
    
    model, tokenizer = load_model(device)
    print(f"Parameters: {model.num_parameters:,}\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() == 'exit':
                print("Goodbye!")
                break
            
            if prompt.lower() == 'clear':
                print("History cleared\n")
                continue
            
            if not prompt:
                continue
            
            response = generate(model, tokenizer, prompt, device)
            print(f"Agent: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()