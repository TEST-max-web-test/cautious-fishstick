import torch
from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
import os

def load_model_and_tokenizer():
    vocab_size = 100
    hidden_size = 64
    num_layers = 2
    num_heads = 2
    ff_expansion = 2
    checkpoint_path = os.path.join(os.path.dirname(__file__), '../utils/checkpoint.pt')
    bpe_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bpe.model'))
    if not os.path.exists(bpe_model_path):
        bpe_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../bpe.model'))
    tokenizer = CustomTokenizer(bpe_model_path)
    model = CustomTransformer(vocab_size, hidden_size, num_layers, num_heads, ff_expansion)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer

def chat():
    model, tokenizer = load_model_and_tokenizer()
    print("Chat with your custom agent! Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.strip().lower() == 'exit':
            print("Exiting chat.")
            break
        ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([ids], dtype=torch.long)
        response_ids = ids.copy()
        max_response_length = 50
        temperature = 1.0
        for _ in range(max_response_length):
            with torch.no_grad():
                logits = model(torch.tensor([response_ids], dtype=torch.long))
                logits = logits[0, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            # Stop if EOS token (if defined, e.g. 0) is generated
            if next_id == 0:
                break
            response_ids.append(next_id)
        # Only show generated part, not echo
        generated = response_ids[len(ids):]
        response = tokenizer.decode(generated) if generated else "(No response generated)"
        print("Agent:", response)

if __name__ == "__main__":
    chat()