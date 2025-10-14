#!/usr/bin/env python3
"""
FIXED CHAT INTERFACE with robust generation
Addresses: tokenizer consistency, better sampling, repetition prevention
"""
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
        'ai_cybersec_custom/train/HERE/checkpoint.pt',
        os.path.join(script_dir, 'train/HERE/checkpoint.pt'),
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


def generate_greedy(model, tokenizer, prompt, device='cpu', max_len=100):
    """
    Greedy decoding - always pick most likely token
    This is deterministic and helps debug if the model learned anything
    """
    formatted = f"User: {prompt}\nAgent:"
    
    # Encode once and save the length
    input_ids = tokenizer.encode(formatted, add_bos=True)
    prompt_length = len(input_ids)
    
    ids = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([ids], device=device)
            
            # Truncate if too long
            if x.size(1) > MODEL_CONFIG['seq_len']:
                x = x[:, -MODEL_CONFIG['seq_len']:]
            
            # Forward
            logits, _ = model(x)
            next_logits = logits[0, -1]
            
            # Greedy: pick most likely
            next_token = next_logits.argmax().item()
            
            # Stop conditions
            if next_token in (tokenizer.EOS, tokenizer.PAD):
                break
            
            ids.append(next_token)
    
    # Decode only the response part
    response_ids = ids[prompt_length:]
    response = tokenizer.decode(response_ids).strip()
    
    return response


def generate_with_penalties(model, tokenizer, prompt, device='cpu'):
    """
    Advanced generation with repetition penalty and better sampling
    """
    max_len = INFER_CONFIG['max_response_length']
    temp = INFER_CONFIG['temperature']
    top_k = INFER_CONFIG['top_k']
    top_p = INFER_CONFIG.get('top_p', 0.92)
    rep_penalty = INFER_CONFIG.get('repetition_penalty', 1.2)
    
    formatted = f"User: {prompt}\nAgent:"
    
    # Encode once and save the length
    input_ids = tokenizer.encode(formatted, add_bos=True)
    prompt_length = len(input_ids)
    
    ids = input_ids.copy()
    generated_tokens = []  # Track what we've generated for repetition penalty
    
    with torch.no_grad():
        for step in range(max_len):
            x = torch.tensor([ids], device=device)
            
            # Truncate if too long
            if x.size(1) > MODEL_CONFIG['seq_len']:
                x = x[:, -MODEL_CONFIG['seq_len']:]
            
            # Forward
            logits, _ = model(x)
            next_logits = logits[0, -1].clone()
            
            # Apply repetition penalty
            if generated_tokens and rep_penalty != 1.0:
                for token_id in set(generated_tokens):
                    # If the token has been generated, reduce its probability
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= rep_penalty
                    else:
                        next_logits[token_id] *= rep_penalty
            
            # Apply temperature
            next_logits = next_logits / temp
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, min(top_k, len(next_logits)))[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            
            # Prevent NaN
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("⚠️ NaN/Inf in probs, using greedy")
                next_token = next_logits.argmax().item()
            else:
                next_token = torch.multinomial(probs, 1).item()
            
            # Stop conditions
            if next_token in (tokenizer.EOS, tokenizer.PAD):
                break
            
            # Check for degenerate output (too many UNKs or special tokens)
            if next_token == tokenizer.UNK:
                if len(generated_tokens) > 5 and generated_tokens[-3:].count(tokenizer.UNK) >= 2:
                    break  # Too many UNKs in a row
            
            ids.append(next_token)
            generated_tokens.append(next_token)
            
            # Emergency stop for repetitive output
            if len(generated_tokens) > 10:
                last_5 = generated_tokens[-5:]
                if len(set(last_5)) <= 2:  # Only 1-2 unique tokens in last 5
                    break
    
    # Decode only the response part
    response_ids = ids[prompt_length:]
    response = tokenizer.decode(response_ids).strip()
    
    # Clean up common artifacts
    if "User:" in response:
        response = response.split("User:")[0].strip()
    
    return response


def main():
    print("="*70)
    print("💬 FIXED CHAT INTERFACE")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("Commands:")
    print("  'exit' - quit")
    print("  'clear' - reset")
    print("  'greedy' - toggle greedy decoding (for debugging)")
    print("="*70 + "\n")
    
    model, tokenizer = load_model(device)
    print(f"Parameters: {model.num_parameters:,}")
    print(f"Vocab: {tokenizer.vocab_size()}")
    print(f"Special tokens: PAD={tokenizer.PAD} UNK={tokenizer.UNK} BOS={tokenizer.BOS} EOS={tokenizer.EOS}\n")
    
    use_greedy = False
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() == 'exit':
                print("Goodbye!")
                break
            
            if prompt.lower() == 'clear':
                print("History cleared\n")
                continue
            
            if prompt.lower() == 'greedy':
                use_greedy = not use_greedy
                mode = "greedy (deterministic)" if use_greedy else "sampling (with penalties)"
                print(f"Switched to {mode} mode\n")
                continue
            
            if not prompt:
                continue
            
            # Generate response
            if use_greedy:
                response = generate_greedy(model, tokenizer, prompt, device)
            else:
                response = generate_with_penalties(model, tokenizer, prompt, device)
            
            if not response or len(response) < 3:
                print(f"Agent: [model produced empty/invalid response]\n")
            else:
                print(f"Agent: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()