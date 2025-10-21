#!/usr/bin/env python3
"""
Convert filtered JSONL data to plain text corpus
Creates a simple text file with all content for training
"""

import json
from pathlib import Path

INPUT_FILE = Path("filtered_data/consolidated_training_data.jsonl")
OUTPUT_FILE = Path("filtered_data/text_corpus.txt")

def main():
    print("="*80)
    print("📝 CONVERTING FILTERED DATA TO TEXT CORPUS")
    print("="*80)
    
    total_chars = 0
    item_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                data = json.loads(line)
                content = data.get('content', '')
                
                # Add metadata header
                source = data.get('source', 'unknown')
                title = data.get('title', data.get('file', 'Untitled'))
                
                outfile.write(f"\n{'='*80}\n")
                outfile.write(f"SOURCE: {source}\n")
                outfile.write(f"TITLE: {title}\n")
                outfile.write(f"{'='*80}\n\n")
                outfile.write(content)
                outfile.write(f"\n\n")
                
                total_chars += len(content)
                item_count += 1
                
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Items processed: {item_count}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Output file: {OUTPUT_FILE}")
    print(f"   File size: {size_mb:.2f} MB")
    print("\n" + "="*80)
    print("📚 DATASET INFORMATION")
    print("="*80)
    print("\nTwo training data formats available:")
    print("\n1. JSONL Format (structured):")
    print("   - filtered_data/consolidated_training_data.jsonl")
    print("   - Good for: Fine-tuning, RAG systems, structured learning")
    print("   - Format: One JSON object per line with metadata")
    print("\n2. Plain Text Format (simple):")
    print("   - filtered_data/text_corpus.txt")
    print("   - Good for: Language model pre-training, simple text analysis")
    print("   - Format: Plain text with section headers")
    print("\n3. Q&A Format (conversational):")
    print("   - corpus.txt (existing)")
    print("   - Good for: Conversational AI training")
    print("   - Format: User question / Agent answer pairs")
    print("="*80)

if __name__ == "__main__":
    main()
