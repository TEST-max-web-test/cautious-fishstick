# Custom Transformer-based Language Model (PyTorch)

This project implements a fully custom transformer-based language model from scratch using PyTorch, inspired by best practices from GPT-5 pro, Claude 4.5, and Gemini 2.5 pro.

## Features
- Custom tokenizer (SentencePiece, Unigram/BPE)
- Transformer architecture: rotary positional embeddings, multi-head attention, residual connections, layer normalization, GELU activation
- Configurable depth, width, attention heads, feedforward expansion
- Training loop: AdamW, cosine scheduler, gradient clipping, mixed precision (FP16)
- Dataset loader: raw text/JSONL, tokenization, batching
- Evaluation: perplexity, token accuracy, loss
- Checkpointing: save/load weights
- Modular, scalable, reproducible

## Structure
- `tokenizer/` — Custom tokenizer
- `model/` — Transformer architecture
- `train/` — Training loop
- `data/` — Dataset loader
- `eval/` — Evaluation metrics
- `utils/` — Checkpointing, config, logging

## Usage
1. Install dependencies: `pip install torch sentencepiece`
2. Prepare your dataset in `data/`
3. Train: `python train/train.py`
4. Evaluate: `python eval/evaluate.py`

---
