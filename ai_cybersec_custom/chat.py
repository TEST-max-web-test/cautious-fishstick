#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F

# Ensure parent directories are on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))
sys.path.insert(0, os.path.dirname(os.path.join(script_dir, '..')))

from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, INFER_CONFIG


def load_model_and_tokenizer(device: str = 'cpu'):
    """Load model and tokenizer from checkpoint."""
    
    # Find tokenizer
    possible_tokenizer_paths = [
        os.path.join(script_dir, 'tokenizer', 'bpe.model'),
        os.path.join(script_dir, '..', 'tokenizer', 'bpe.model'),
        'ai_cybersec_custom/tokenizer/bpe.model',
    ]

    tokenizer_path = None
    for path in possible_tokenizer_paths:
        if os.path.exists(path):
            tokenizer_path = path
            break

    if not tokenizer_path:
        print("❌ ERROR: Could not find bpe.model")
        raise FileNotFoundError("bpe.model not found!")

    print(f"📂 Loading tokenizer from: {tokenizer_path}")
    tokenizer = CustomTokenizer(tokenizer_path)

    # Instantiate model
    model = CustomTransformer(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        ff_expansion=MODEL_CONFIG['ff_expansion'],
        dropout=MODEL_CONFIG.get('dropout', 0.1),
        max_seq_len=MODEL_CONFIG['seq_len']
    )

    # Find checkpoint
    possible_checkpoint_paths = [
        'ai_cybersec_custom/train/utils/checkpoint.pt',
        os.path.join(script_dir, 'train', 'utils', 'checkpoint.pt'),
        os.path.join(script_dir, 'utils', 'checkpoint.pt'),
    ]

    checkpoint_path = None
    for path in possible_checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path:
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'], strict=False)
            if 'val_ppl' in state:
                print(f"   Validation PPL: {state['val_ppl']:.2f}")
        else:
            model.load_state_dict(state, strict=False)
        
        print(f"✅ Loaded checkpoint")
    else:
        print("⚠️  No checkpoint found! Using untrained model.")

    model = model.to(device)
    model.eval()
    return model, tokenizer


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list, penalty: float = 1.2):
    """
    Apply repetition penalty to discourage repeating tokens.
    Higher penalty = less repetition
    """
    if penalty != 1.0 and len(generated_ids) > 0:
        for token_id in set(generated_ids):
            # If token has been generated before, reduce its probability
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits


def top_p_top_k_sampling(
    logits: torch.Tensor, 
    generated_ids: list,
    top_p: float = 0.95, 
    top_k: int = 40, 
    temperature: float = 1.0,
    repetition_penalty: float = 1.2
) -> int:
    """
    Advanced sampling with:
    - Temperature control
    - Top-k filtering
    - Top-p (nucleus) sampling
    - Repetition penalty
    """
    
    # Apply temperature
    logits = logits / max(temperature, 1e-8)
    
    # Apply repetition penalty
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
    
    # Top-k filtering
    if top_k > 0:
        kth_vals = torch.topk(logits, min(top_k, len(logits)))[0][..., -1, None]
        indices_to_remove = logits < kth_vals
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    # Top-p (nucleus) filtering
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 0] = False  # Keep at least one token
    
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))
    
    # Sample from the filtered distribution
    probs = F.softmax(sorted_logits, dim=-1)
    next_idx = torch.multinomial(probs, num_samples=1)
    next_token = sorted_indices[next_idx].item()
    
    return next_token


def generate_response(
    model: torch.nn.Module, 
    tokenizer: CustomTokenizer, 
    prompt: str, 
    device: str = 'cpu'
) -> str:
    """Generate response with advanced reasoning and varied output."""
    
    max_length = INFER_CONFIG.get('max_response_length', 100)
    temperature = INFER_CONFIG.get('temperature', 0.8)
    top_p = INFER_CONFIG.get('top_p', 0.92)
    top_k = INFER_CONFIG.get('top_k', 50)
    repetition_penalty = INFER_CONFIG.get('repetition_penalty', 1.2)

    # Format prompt in conversational style
    formatted_prompt = f"User: {prompt}\nAgent:"

    # Encode prompt
    ids = tokenizer.encode(formatted_prompt, add_bos=True)
    response_ids = ids.copy()
    generated_ids = []  # Track generated tokens for repetition penalty

    with torch.no_grad():
        for step in range(max_length):
            input_tensor = torch.tensor([response_ids], dtype=torch.long, device=device)

            # Truncate if too long
            if input_tensor.size(1) > MODEL_CONFIG['seq_len']:
                input_tensor = input_tensor[:, -MODEL_CONFIG['seq_len']:]

            # Forward pass
            logits, _ = model(input_tensor)
            
            # Sample next token
            next_token = top_p_top_k_sampling(
                logits[0, -1],
                generated_ids,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )

            # Stop if we hit end tokens
            if next_token == tokenizer.EOS or next_token == tokenizer.PAD:
                break

            response_ids.append(int(next_token))
            generated_ids.append(int(next_token))

    # Decode response
    generated = response_ids[len(ids):]
    
    if generated:
        try:
            response = tokenizer.decode(generated)
            response = response.strip()
            
            # Stop at next "User:" if present
            if "User:" in response:
                response = response.split("User:")[0].strip()
            
            # Stop at newlines that seem like conversation breaks
            if "\n\n" in response:
                response = response.split("\n\n")[0].strip()
                
        except Exception as e:
            print(f"⚠️  Decode error: {e}")
            response = "(Error decoding response)"
    else:
        response = "(No response generated)"

    return response


def chat():
    """Interactive chat loop with advanced reasoning model."""
    
    print("\n" + "="*70)
    print("🧠 ADVANCED REASONING CHAT AGENT - CYBERSECURITY ASSISTANT")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📍 Device: {device}")
    print("💬 Type 'exit' to quit, 'clear' to clear history")
    print("💬 Type 'temp X' to set temperature (0.1-2.0)")
    print("="*70 + "\n")

    try:
        model, tokenizer = load_model_and_tokenizer(device=device)
        
        print(f"✅ Model ready")
        print(f"   Parameters: {model.num_parameters:,}")
        print(f"   Vocab size: {tokenizer.vocab_size()}")
        print(f"   Architecture: {MODEL_CONFIG['num_layers']}L x {MODEL_CONFIG['hidden_size']}D")
        print(f"   Temperature: {INFER_CONFIG['temperature']}")
        print(f"   Repetition penalty: {INFER_CONFIG['repetition_penalty']}\n")
        
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print("\nPlease ensure:")
        print("  1. Tokenizer trained: python ai_cybersec_custom/tokenizer/train_tokenizer.py")
        print("  2. Model trained: cd ai_cybersec_custom/train && python train.py")
        import traceback
        traceback.print_exc()
        return

    conversation_history = []
    current_temp = INFER_CONFIG['temperature']

    while True:
        try:
            prompt = input("👤 You: ").strip()

            if prompt.lower() == 'exit':
                print("\n👋 Goodbye! Stay secure!")
                break

            if prompt.lower() == 'clear':
                conversation_history = []
                print("🗑️  Conversation history cleared\n")
                continue
            
            # Handle temperature adjustment
            if prompt.lower().startswith('temp '):
                try:
                    new_temp = float(prompt.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        INFER_CONFIG['temperature'] = new_temp
                        current_temp = new_temp
                        print(f"🌡️  Temperature set to {new_temp}\n")
                    else:
                        print("⚠️  Temperature must be between 0.1 and 2.0\n")
                except:
                    print("⚠️  Usage: temp 0.8\n")
                continue

            if not prompt:
                print("⚠️  Please enter a message\n")
                continue

            conversation_history.append(("user", prompt))

            print("🤔 Thinking...", end=" ", flush=True)
            response = generate_response(model, tokenizer, prompt, device=device)
            print("\r" + " " * 50 + "\r", end="")  # Clear "Thinking..."
            print(f"🤖 Agent: {response}\n")

            conversation_history.append(("agent", response))

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing...\n")


def main():
    """Entry point."""
    chat()


if __name__ == "__main__":
    main()