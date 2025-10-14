#!/usr/bin/env python3
"""
Debug script to understand what's happening during generation
Run: python3 debug_generation.py
"""
import torch
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_cybersec_custom.model.custom_transformer import CustomTransformer
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG

def debug_generation():
    print("="*80)
    print("🔍 GENERATION DEBUG")
    print("="*80)
    
    device = 'cpu'
    
    # Load tokenizer
    tokenizer_path = 'ai_cybersec_custom/tokenizer/bpe.model'
    tokenizer = CustomTokenizer(tokenizer_path)
    
    print(f"\n📝 Tokenizer Info:")
    print(f"   Vocab size: {tokenizer.vocab_size()}")
    print(f"   PAD={tokenizer.PAD}, UNK={tokenizer.UNK}, BOS={tokenizer.BOS}, EOS={tokenizer.EOS}")
    
    # Load model
    model = CustomTransformer(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        ff_expansion=MODEL_CONFIG['ff_expansion'],
        dropout=MODEL_CONFIG['dropout'],
        max_seq_len=MODEL_CONFIG['seq_len']
    ).to(device)
    
    checkpoint_path = 'ai_cybersec_custom/train/HERE/checkpoint.pt'
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"✅ Loaded checkpoint\n")
    else:
        print("❌ No checkpoint found!\n")
        return
    
    model.eval()
    
    # Test prompt
    prompt = "What is SQL injection?"
    
    print("="*80)
    print("TEST 1: Tokenizer Consistency Check")
    print("="*80)
    
    formatted = f"User: {prompt}\nAgent:"
    
    # Encode multiple times to check consistency
    ids1 = tokenizer.encode(formatted, add_bos=True)
    ids2 = tokenizer.encode(formatted, add_bos=True)
    ids3 = tokenizer.encode(formatted, add_bos=True)
    
    print(f"Formatted text: {repr(formatted)}")
    print(f"\nEncoding 1: {ids1}")
    print(f"Encoding 2: {ids2}")
    print(f"Encoding 3: {ids3}")
    
    if ids1 == ids2 == ids3:
        print("✅ Tokenizer is consistent")
    else:
        print("❌ TOKENIZER IS INCONSISTENT! This is a critical bug!")
        return
    
    print(f"\nDecoded: {repr(tokenizer.decode(ids1))}")
    
    # Check what training samples look like
    print("\n" + "="*80)
    print("TEST 2: Training Sample Format Check")
    print("="*80)
    
    training_example = "User: What is SQL injection?\nAgent: SQL injection inserts malicious SQL code into input fields to manipulate database queries."
    train_ids = tokenizer.encode(training_example, add_bos=True, add_eos=True)
    
    print(f"Training sample: {repr(training_example[:80])}...")
    print(f"Token IDs: {train_ids[:20]}... (length: {len(train_ids)})")
    print(f"Decoded: {repr(tokenizer.decode(train_ids)[:80])}...")
    
    # Now test generation step by step
    print("\n" + "="*80)
    print("TEST 3: Step-by-Step Generation Debug")
    print("="*80)
    
    ids = tokenizer.encode(formatted, add_bos=True)
    print(f"\nInitial prompt IDs: {ids}")
    print(f"Initial prompt text: {repr(tokenizer.decode(ids))}")
    
    with torch.no_grad():
        for step in range(20):  # Only 20 steps for debugging
            x = torch.tensor([ids], device=device)
            
            print(f"\n--- Step {step+1} ---")
            print(f"Input shape: {x.shape}")
            print(f"Input IDs: {x[0].tolist()}")
            
            # Forward pass
            logits, _ = model(x)
            next_logits = logits[0, -1]
            
            print(f"Logits shape: {logits.shape}")
            print(f"Next token logits (last position): min={next_logits.min():.2f}, max={next_logits.max():.2f}, mean={next_logits.mean():.2f}")
            
            # Get top 10 predictions
            probs = F.softmax(next_logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=10)
            
            print(f"\nTop 10 predictions:")
            for i, (prob, tid) in enumerate(zip(top_probs, top_ids)):
                token_text = tokenizer.decode([tid.item()])
                print(f"  {i+1}. ID={tid.item():4d} prob={prob:.4f} text={repr(token_text)}")
            
            # Sample with temperature
            temp = 0.8
            scaled_logits = next_logits / temp
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            print(f"\nSampled token: ID={next_token}, text={repr(tokenizer.decode([next_token]))}")
            
            # Check if it's a special token
            if next_token == tokenizer.PAD:
                print("⚠️ Generated PAD token - stopping")
                break
            elif next_token == tokenizer.EOS:
                print("✅ Generated EOS token - stopping")
                break
            elif next_token == tokenizer.UNK:
                print("⚠️ Generated UNK token")
            
            ids.append(next_token)
            
            # Show current generated text
            current_text = tokenizer.decode(ids)
            print(f"\nCurrent full text: {repr(current_text)}")
    
    # Final result
    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    
    response_start = len(tokenizer.encode(formatted, add_bos=True))
    response_ids = ids[response_start:]
    response = tokenizer.decode(response_ids).strip()
    
    print(f"Response IDs: {response_ids}")
    print(f"Response text: {repr(response)}")
    
    # Check for common issues
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    issues = []
    
    # Check if model is producing mostly PAD/UNK tokens
    special_count = sum(1 for tid in response_ids if tid in [tokenizer.PAD, tokenizer.UNK, tokenizer.BOS, tokenizer.EOS])
    if special_count > len(response_ids) * 0.5:
        issues.append(f"⚠️ {special_count}/{len(response_ids)} tokens are special tokens")
    
    # Check if response is empty or very short
    if len(response_ids) < 5:
        issues.append(f"⚠️ Response is very short ({len(response_ids)} tokens)")
    
    # Check if output is repetitive
    if len(response_ids) > 3:
        unique_ratio = len(set(response_ids)) / len(response_ids)
        if unique_ratio < 0.3:
            issues.append(f"⚠️ Output is very repetitive (unique ratio: {unique_ratio:.2f})")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ No obvious issues detected (but output may still be wrong)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    debug_generation()