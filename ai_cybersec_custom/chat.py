import torch
from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
import os

from ai_cybersec_custom.utils.config import MODEL_CONFIG, INFER_CONFIG, TRAIN_CONFIG
def load_model_and_tokenizer():
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

def chat():
    model, tokenizer = load_model_and_tokenizer()
    print("Chat with your custom agent! Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.strip().lower() == 'exit':
            print("Exiting chat.")
            break
        # Batched generation: support multiple prompts (for now, just one)
        prompts = [prompt]
        batch_ids = [tokenizer.encode(p, add_bos=True, add_eos=True) for p in prompts]
        max_response_length = INFER_CONFIG['max_response_length']
        temperature = INFER_CONFIG['temperature']
        top_p = INFER_CONFIG['top_p']
        # Speculative decoding stub (single batch)
        responses = []
        for ids in batch_ids:
            response_ids = ids.copy()
            kv_cache = None
            for _ in range(max_response_length):
                input_tensor = torch.tensor([response_ids], dtype=torch.long)
                with torch.no_grad():
                    logits, _ = model(input_tensor)
                    logits = logits[0, -1] / temperature
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    cutoff = cumulative_probs > top_p
                    if cutoff.any():
                        cutoff_idx = torch.where(cutoff)[0][0]
                        sorted_logits = sorted_logits[:cutoff_idx+1]
                        sorted_indices = sorted_indices[:cutoff_idx+1]
                    probs = torch.softmax(sorted_logits, dim=-1)
                    next_id = sorted_indices[torch.multinomial(probs, num_samples=1)].item()
                # Stop if EOS token (50257) is generated
                if next_id == tokenizer.EOS:
                    break
                response_ids.append(next_id)
            generated = response_ids[len(ids):]
            response = tokenizer.decode(generated) if generated else "(No response generated)"
            responses.append(response)
        print("Agent:", responses[0])

if __name__ == "__main__":
    chat()