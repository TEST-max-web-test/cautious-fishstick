import torch
from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, INFER_CONFIG, TRAIN_CONFIG
import os


def load_model_and_tokenizer(device: str = 'cpu'):
    """Load model and tokenizer from checkpoint."""
    # ✅ FIXED: Try multiple paths for tokenizer
    possible_tokenizer_paths = [
        'ai_cybersec_custom/tokenizer/bpe.model',
        'bpe.model',
        os.path.join(os.path.dirname(__file__), 'tokenizer/bpe.model'),
        os.path.join(os.path.dirname(__file__), '../bpe.model'),
    ]
    
    tokenizer_path = None
    for path in possible_tokenizer_paths:
        if os.path.exists(path):
            tokenizer_path = path
            break
    
    if not tokenizer_path:
        print("❌ ERROR: Could not find bpe.model in any of these locations:")
        for path in possible_tokenizer_paths:
            print(f"   - {os.path.abspath(path)}")
        raise FileNotFoundError("bpe.model not found!")
    
    print(f"📂 Loading tokenizer from: {tokenizer_path}")
    tokenizer = CustomTokenizer(tokenizer_path)
    
    # Load model
    model = CustomTransformer(
        MODEL_CONFIG['vocab_size'],
        MODEL_CONFIG['hidden_size'],
        MODEL_CONFIG['num_layers'],
        MODEL_CONFIG['num_heads'],
        MODEL_CONFIG['ff_expansion']
    )
    
    # Load checkpoint
    checkpoint_path = TRAIN_CONFIG['checkpoint_path']
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}")
        print(f"   Model will use random weights (not trained)")
    
    model = model.to(device)
    model.eval()
    return model, tokenizer


def top_p_top_k_sampling(logits: torch.Tensor, top_p: float = 0.95, 
                         top_k: int = 40, temperature: float = 1.0) -> int:
    """Apply combined top-p and top-k sampling."""
    logits = logits / temperature
    
    # Top-k
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Top-p
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 0] = False
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    probs = torch.softmax(sorted_logits + 1e-10, dim=-1)
    next_idx = torch.multinomial(probs, num_samples=1)
    next_token = sorted_indices[next_idx].item()
    
    return next_token


def generate_response(model: torch.nn.Module, tokenizer: CustomTokenizer, 
                     prompt: str, device: str = 'cpu') -> str:
    """Generate response from prompt using the model."""
    max_length = INFER_CONFIG['max_response_length']
    temperature = INFER_CONFIG['temperature']
    top_p = INFER_CONFIG['top_p']
    top_k = INFER_CONFIG.get('top_k', 40)
    
    # Format prompt properly
    formatted_prompt = f"User: {prompt}\nAgent:"
    
    # Encode prompt
    ids = tokenizer.encode(formatted_prompt, add_bos=True)
    response_ids = ids.copy()
    
    # Generate tokens autoregressively
    with torch.no_grad():
        for step in range(max_length):
            input_tensor = torch.tensor([response_ids], dtype=torch.long, device=device)
            
            # Truncate if too long
            if input_tensor.size(1) > MODEL_CONFIG['seq_len']:
                input_tensor = input_tensor[:, -MODEL_CONFIG['seq_len']:]
            
            logits, _ = model(input_tensor)
            next_token = top_p_top_k_sampling(
                logits[0, -1], 
                top_p=top_p, 
                top_k=top_k,
                temperature=temperature
            )
            
            # Stop conditions
            if next_token == tokenizer.EOS:
                break
            if next_token == tokenizer.PAD:
                break
            
            response_ids.append(next_token)
    
    # Decode response (exclude prompt)
    generated = response_ids[len(ids):]
    if generated:
        try:
            response = tokenizer.decode(generated)
            # Clean up response
            response = response.strip()
            # Stop at "User:" if model generates next turn
            if "User:" in response:
                response = response.split("User:")[0].strip()
        except Exception as e:
            print(f"⚠️  Decode error: {e}")
            response = "(Error decoding response)"
    else:
        response = "(No response generated)"
    
    return response


def chat():
    """Interactive chat loop with the model."""
    print("\n" + "="*70)
    print("🤖 CUSTOM TRANSFORMER CHAT AGENT - CYBERSECURITY ASSISTANT")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📍 Device: {device}")
    print("💬 Type 'exit' to quit, 'clear' to clear history")
    print("="*70 + "\n")
    
    try:
        model, tokenizer = load_model_and_tokenizer(device=device)
        print(f"✅ Model ready | Parameters: {model.num_parameters:,}")
        print(f"   Vocab size: {tokenizer.vocab_size()}")
        print(f"   Special tokens: PAD={tokenizer.PAD}, BOS={tokenizer.BOS}, EOS={tokenizer.EOS}\n")
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print("\nPlease ensure:")
        print("  1. Tokenizer trained: python ai_cybersec_custom/tokenizer/train_tokenizer.py")
        print("  2. Model trained: PYTHONPATH=./ python ai_cybersec_custom/train/train.py")
        return
    
    conversation_history = []
    
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
            
            if not prompt:
                print("⚠️  Please enter a message\n")
                continue
            
            # Add to history
            conversation_history.append(("user", prompt))
            
            # Generate response
            print("🤔 Generating response...", end=" ", flush=True)
            response = generate_response(model, tokenizer, prompt, device=device)
            print("\r" + " " * 50 + "\r", end="")  # Clear "Generating..." line
            print(f"🤖 Agent: {response}\n")
            
            # Add response to history
            conversation_history.append(("agent", response))
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Continuing...\n")


def main():
    """Entry point."""
    chat()


if __name__ == "__main__":
    main()