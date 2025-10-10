import torch
from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, INFER_CONFIG, TRAIN_CONFIG
import os


def load_model_and_tokenizer(device: str = 'cpu'):
    """Load model and tokenizer from checkpoint."""
    checkpoint_path = os.path.join(os.path.dirname(__file__), '../' + TRAIN_CONFIG['checkpoint_path'])
    bpe_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bpe.model'))
    if not os.path.exists(bpe_model_path):
        bpe_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../bpe.model'))
    
    tokenizer = CustomTokenizer(bpe_model_path)
    model = CustomTransformer(
        MODEL_CONFIG['vocab_size'],
        MODEL_CONFIG['hidden_size'],
        MODEL_CONFIG['num_layers'],
        MODEL_CONFIG['num_heads'],
        MODEL_CONFIG['ff_expansion']
    )
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    return model, tokenizer


def top_p_top_k_sampling(logits: torch.Tensor, top_p: float = 0.95, 
                         top_k: int = 40, temperature: float = 1.0) -> int:
    """
    Apply combined top-p and top-k sampling.
    
    Args:
        logits: Model output logits for last token
        top_p: Nucleus sampling threshold (0.95 = keep top 95%)
        top_k: Keep only top k most likely tokens
        temperature: Softness of distribution (lower = more deterministic)
    
    Returns:
        next_token: Selected token ID
    """
    logits = logits / temperature
    
    # Top-k: keep only top k logits
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Top-p: keep only top p% of probability mass
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Find cutoff index for top-p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 0] = False  # Keep at least one token
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    # Add small epsilon for numerical stability
    probs = torch.softmax(sorted_logits + 1e-10, dim=-1)
    next_idx = torch.multinomial(probs, num_samples=1)
    next_token = sorted_indices[next_idx].item()
    
    return next_token


def generate_response(model: torch.nn.Module, tokenizer: CustomTokenizer, 
                     prompt: str, device: str = 'cpu') -> str:
    """
    Generate response from prompt using the model.
    
    Args:
        model: Language model
        tokenizer: Tokenizer instance
        prompt: Input text
        device: Device to run on
    
    Returns:
        response: Generated text
    """
    max_length = INFER_CONFIG['max_response_length']
    temperature = INFER_CONFIG['temperature']
    top_p = INFER_CONFIG['top_p']
    top_k = INFER_CONFIG.get('top_k', 40)
    
    # Encode prompt
    ids = tokenizer.encode(prompt, add_bos=True)
    response_ids = ids.copy()
    
    # Generate tokens autoregressively
    with torch.no_grad():
        for step in range(max_length):
            input_tensor = torch.tensor([response_ids], dtype=torch.long, device=device)
            logits, _ = model(input_tensor)
            next_token = top_p_top_k_sampling(
                logits[0, -1], 
                top_p=top_p, 
                top_k=top_k,
                temperature=temperature
            )
            
            # Stop if EOS token is generated
            if next_token == tokenizer.EOS:
                break
            
            response_ids.append(next_token)
    
    # Decode response (exclude prompt)
    generated = response_ids[len(ids):]
    response = tokenizer.decode(generated) if generated else "(No response generated)"
    return response


def chat():
    """Interactive chat loop with the model."""
    print("\n" + "="*70)
    print("🤖 CUSTOM TRANSFORMER CHAT AGENT")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📍 Device: {device}")
    print("💬 Type 'exit' to quit, 'clear' to clear history")
    print("="*70 + "\n")
    
    model, tokenizer = load_model_and_tokenizer(device=device)
    print(f"✅ Model ready | Parameters: {model.num_parameters:,}\n")
    
    conversation_history = []
    
    while True:
        prompt = input("👤 You: ").strip()
        
        if prompt.lower() == 'exit':
            print("\n👋 Goodbye!")
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
        print("\n")
        print(f"🤖 Agent: {response}\n")
        
        # Add response to history
        conversation_history.append(("agent", response))


def main():
    """Entry point."""
    chat()


if __name__ == "__main__":
    main()