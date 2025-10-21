#!/usr/bin/env python3
"""
Combine existing corpus.txt with filtered_data into one comprehensive training corpus
This creates a single file with ALL training data for the model
"""

import json
from pathlib import Path

# Input files
CORPUS_FILE = Path("corpus.txt")
CONSOLIDATED_FILE = Path("filtered_data/consolidated_training_data.jsonl")
OUTPUT_FILE = Path("combined_corpus.txt")

def main():
    print("="*80)
    print("🔗 COMBINING ALL TRAINING DATA INTO SINGLE CORPUS")
    print("="*80)
    
    total_chars = 0
    total_items = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # First, copy the existing Q&A corpus
        print("\n📋 Step 1: Adding existing Q&A corpus...")
        if CORPUS_FILE.exists():
            with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
                corpus_content = f.read()
                outfile.write(corpus_content)
                
                # Count Q&A pairs (separated by double newlines)
                qa_pairs = [c for c in corpus_content.split('\n\n') if c.strip()]
                total_items += len(qa_pairs)
                total_chars += len(corpus_content)
                
                print(f"   ✅ Added {len(qa_pairs)} Q&A conversation pairs")
                print(f"   📏 Size: {len(corpus_content):,} characters")
                
                # Add separator if corpus doesn't end with double newline
                if not corpus_content.endswith('\n\n'):
                    outfile.write('\n\n')
        else:
            print(f"   ⚠️  corpus.txt not found, skipping...")
        
        # Then, add all the filtered documents
        print("\n📚 Step 2: Adding filtered security documents...")
        if CONSOLIDATED_FILE.exists():
            doc_count = 0
            doc_chars = 0
            
            with open(CONSOLIDATED_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        content = data.get('content', '').strip()
                        
                        if not content:
                            continue
                        
                        # Add source metadata as a header (optional but helpful)
                        source = data.get('source', 'unknown')
                        repo = data.get('repo', '')
                        title = data.get('title', data.get('file', 'Document'))
                        
                        # Write document with metadata header
                        outfile.write(f"Document: {title}\n")
                        if repo:
                            outfile.write(f"Source: {repo}\n")
                        outfile.write(f"{'-'*60}\n")
                        outfile.write(content)
                        outfile.write('\n\n')
                        
                        doc_count += 1
                        doc_chars += len(content)
                        
                        if doc_count % 100 == 0:
                            print(f"   📄 Processed {doc_count} documents...")
                        
                    except Exception as e:
                        print(f"   ⚠️  Error processing line: {e}")
                        continue
            
            total_items += doc_count
            total_chars += doc_chars
            
            print(f"   ✅ Added {doc_count} security documents")
            print(f"   📏 Size: {doc_chars:,} characters")
        else:
            print(f"   ❌ {CONSOLIDATED_FILE} not found!")
            return
    
    # Summary
    output_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*80)
    print("✅ COMBINED CORPUS CREATED")
    print("="*80)
    print(f"\n📊 Statistics:")
    print(f"   Total items: {total_items:,}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   File size: {output_size:.2f} MB")
    print(f"   Output: {OUTPUT_FILE.absolute()}")
    
    print("\n📝 Content breakdown:")
    qa_percentage = (len(corpus_content) / total_chars * 100) if 'corpus_content' in locals() else 0
    doc_percentage = 100 - qa_percentage
    print(f"   Q&A pairs: {qa_percentage:.1f}%")
    print(f"   Security docs: {doc_percentage:.1f}%")
    
    print("\n" + "="*80)
    print("🎯 READY FOR TRAINING")
    print("="*80)
    print("\nNext steps:")
    print("1. Training script will now use combined_corpus.txt")
    print("2. Run: python ai_cybersec_custom/train/train.py")
    print("="*80)

if __name__ == "__main__":
    main()
