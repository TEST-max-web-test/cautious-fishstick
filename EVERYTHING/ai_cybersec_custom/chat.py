#!/usr/bin/env python3
"""
PRODUCTION CHAT INTERFACE
Uses the modern transformer's built-in generation method
"""
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_cybersec_custom.model.custom_transformer import ModernTransformer
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, INFER_CONFIG


def load_model(device='cpu'):
    """Load trained model and tokenizer"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load tokenizer
    tokenizer_path = os.path.join(script_dir, 'tokenizer/bpe.model')
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print("Run: cd ai_cybersec_custom/tokenizer && python3 train_tokenizer.py")
        sys.exit(1)
    
    tokenizer = CustomTokenizer(tokenizer_path)
    
    # Create model
    model = ModernTransformer(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        ff_expansion=MODEL_CONFIG['ff_expansion'],
        dropout=0.0,  # No dropout during inference
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
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Loaded checkpoint")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Val PPL: {checkpoint.get('val_ppl', 'unknown'):.2f}")
    else:
        print("⚠️  No checkpoint found - using untrained model")
    
    model.eval()
    return model, tokenizer


def chat_loop(model, tokenizer, device):
    """Interactive chat loop"""
    print("\n" + "="*70)
    print("💬 CHAT INTERFACE - Type 'help' for commands")
    print("="*70)
    
    # Current settings
    settings = {
        'temperature': INFER_CONFIG['temperature'],
        'top_p': INFER_CONFIG['top_p'],
        'top_k': INFER_CONFIG['top_k'],
        'repetition_penalty': INFER_CONFIG['repetition_penalty'],
        'max_length': INFER_CONFIG['max_response_length']
    }
    
    print(f"\nSettings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  exit - quit")
                print("  help - show this message")
                print("  settings - show current settings")
                print("  temp <value> - set temperature (0.1-2.0)")
                print("  topp <value> - set top_p (0.1-1.0)")
                print("  penalty <value> - set repetition penalty (1.0-2.0)")
                print()
                continue
            
            if user_input.lower() == 'settings':
                print("\nCurrent settings:")
                for key, value in settings.items():
                    print(f"  {key}: {value}")
                print()
                continue
            
            # Setting adjustments
            if user_input.lower().startswith('temp '):
                try:
                    val = float(user_input.split()[1])
                    settings['temperature'] = max(0.1, min(2.0, val))
                    print(f"Temperature set to {settings['temperature']}\n")
                except:
                    print("Usage: temp <value> (0.1-2.0)\n")
                continue
            
            if user_input.lower().startswith('topp '):
                try:
                    val = float(user_input.split()[1])
                    settings['top_p'] = max(0.1, min(1.0, val))
                    print(f"Top-p set to {settings['top_p']}\n")
                except:
                    print("Usage: topp <value> (0.1-1.0)\n")
                continue
            
            if user_input.lower().startswith('penalty '):
                try:
                    val = float(user_input.split()[1])
                    settings['repetition_penalty'] = max(1.0, min(2.0, val))
                    print(f"Repetition penalty set to {settings['repetition_penalty']}\n")
                except:
                    print("Usage: penalty <value> (1.0-2.0)\n")
                continue
            
            # Generate response
            formatted_prompt = f"User: {user_input}\nAgent:"
            input_ids = torch.tensor(
                [tokenizer.encode(formatted_prompt, add_bos=True)],
                device=device
            )
            
            # Generate using model's built-in method
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=settings['max_length'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    top_p=settings['top_p'],
                    repetition_penalty=settings['repetition_penalty']
                )
            
            # Decode response
            response_ids = output[0, input_ids.size(1):].tolist()
            
            # Remove EOS and PAD tokens
            response_ids = [
                tid for tid in response_ids 
                if tid not in (tokenizer.EOS, tokenizer.PAD)
            ]
            
            response = tokenizer.decode(response_ids).strip()
            
            # Clean up artifacts
            if "User:" in response:
                response = response.split("User:")[0].strip()
            
            if not response or len(response) < 3:
                print("Agent: [empty response - try adjusting temperature]\n")
            else:
                print(f"Agent: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()


def main():
    print("="*70)
    print("🤖 AI CYBERSEC ASSISTANT - PRODUCTION VERSION")
    print("="*70)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model, tokenizer = load_model(device)
    
    print(f"\nModel:")
    print(f"  Parameters: {model.num_parameters:,}")
    print(f"  Vocabulary: {tokenizer.vocab_size()} tokens")
    
    # Start chat
    chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()