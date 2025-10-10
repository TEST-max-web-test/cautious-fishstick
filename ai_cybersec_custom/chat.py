import torch
from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, INFER_CONFIG, TRAIN_CONFIG
import os


def load_model_and_tokenizer():
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
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model, tokenizer


def top_p_sampling(logits, top_p=0.95, temperature=1.0):
    """Apply top-p (nucleus) sampling with temperature."""
    logits = logits / temperature
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


def chat():
    """Interactive chat loop with the model."""
    model, tokenizer = load_model_and_tokenizer()
    print("Chat with your custom agent! Type 'exit' to quit.")
    print("-" * 60)
    
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() == 'exit':
            print("Exiting chat.")
            break
        
        if not prompt:
            continue
        
        # Encode prompt with BOS token
        ids = tokenizer.encode(prompt, add_bos=True)
        max_response_length = INFER_CONFIG['max_response_length']
        temperature = INFER_CONFIG['temperature']
        top_p = INFER_CONFIG['top_p']
        
        response_ids = ids.copy()
        
        # Generate tokens autoregressively
        with torch.no_grad():
            for _ in range(max_response_length):
                input_tensor = torch.tensor([response_ids], dtype=torch.long)
                logits, _ = model(input_tensor)
                next_token = top_p_sampling(logits[0, -1], top_p=top_p, temperature=temperature)
                
                # Stop if EOS token is generated
                if next_token == tokenizer.EOS:
                    break
                
                response_ids.append(next_token)
        
        # Decode and display response
        generated = response_ids[len(ids):]
        response = tokenizer.decode(generated) if generated else "(No response generated)"
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    chat()