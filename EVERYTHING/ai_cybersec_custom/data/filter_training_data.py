#!/usr/bin/env python3
"""
TRAINING DATA FILTER
Filters scraped data to keep only high-quality, useful training data
"""

import json
import re
from pathlib import Path
from collections import Counter

# Configuration
INPUT_DIR = Path("scraped_data")
OUTPUT_DIR = Path("filtered_data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Minimum content length for quality filtering
MIN_CONTENT_LENGTH = 200

# Patterns to exclude (garbage files)
EXCLUDE_PATTERNS = [
    r'\.github',
    r'ISSUE_TEMPLATE',
    r'PULL_REQUEST_TEMPLATE',
    r'\.yml$',
    r'\.yaml$',
    r'CODE_OF_CONDUCT',
    r'CONTRIBUTING',
    r'LICENSE',
    r'README\.md$',
    r'^\.gitignore',
    r'^\.git',
    r'package\.json',
    r'composer\.json',
    r'requirements\.txt',
    r'Gemfile',
    r'Makefile',
    r'Dockerfile',
]

# Keywords that indicate useful security content
USEFUL_KEYWORDS = [
    'vulnerability', 'exploit', 'attack', 'security', 'penetration',
    'hacking', 'injection', 'xss', 'csrf', 'authentication', 'authorization',
    'encryption', 'ctf', 'writeup', 'payload', 'reverse shell', 'privilege escalation',
    'buffer overflow', 'sql injection', 'command injection', 'rce', 'lfi', 'rfi',
    'ssrf', 'deserialization', 'xxe', 'cors', 'owasp', 'cve', 'malware', 
    'ransomware', 'phishing', 'social engineering', 'reconnaissance', 'enumeration',
    'lateral movement', 'persistence', 'evasion', 'defense', 'mitigation',
    'zero-day', 'backdoor', 'trojan', 'rootkit', 'keylogger', 'botnet'
]

def is_garbage_file(file_path, content):
    """Check if file is garbage based on path and content"""
    
    # Check if path matches exclude patterns
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, file_path, re.IGNORECASE):
            return True
    
    # Check if content is too short
    if len(content) < MIN_CONTENT_LENGTH:
        return True
    
    # Check if content has useful security keywords
    content_lower = content.lower()
    keyword_count = sum(1 for keyword in USEFUL_KEYWORDS if keyword in content_lower)
    
    # If no security keywords found, it's likely garbage
    if keyword_count == 0:
        return True
    
    return False

def filter_exploitdb_data(items):
    """Filter ExploitDB data - most entries just have paths, not actual content"""
    filtered = []
    
    for item in items:
        content = item.get('content', '')
        
        # Skip if it's just metadata without real exploit content
        if len(content) < 300:
            continue
        
        # Skip if it's just a file path reference
        if content.count('\n') < 5:
            continue
        
        filtered.append(item)
    
    return filtered

def filter_github_data(items):
    """Filter GitHub data to remove templates and config files"""
    filtered = []
    
    for item in items:
        file_path = item.get('file', '')
        content = item.get('content', '')
        
        # Skip garbage files
        if is_garbage_file(file_path, content):
            continue
        
        # Skip very short files
        if len(content) < MIN_CONTENT_LENGTH:
            continue
        
        filtered.append(item)
    
    return filtered

def filter_blog_data(items):
    """Filter blog data to keep only substantive articles"""
    filtered = []
    
    for item in items:
        content = item.get('content', '')
        
        # Must have minimum length
        if len(content) < 500:
            continue
        
        # Must have some security-related content
        content_lower = content.lower()
        keyword_count = sum(1 for keyword in USEFUL_KEYWORDS if keyword in content_lower)
        
        if keyword_count < 2:
            continue
        
        filtered.append(item)
    
    return filtered

def process_file(input_file):
    """Process a single JSONL file"""
    if not input_file.exists() or input_file.stat().st_size == 0:
        print(f"⏭️  Skipping empty file: {input_file.name}")
        return None
    
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except:
                continue
    
    if not items:
        print(f"⏭️  Skipping file with no valid items: {input_file.name}")
        return None
    
    source_type = items[0].get('source', 'unknown')
    original_count = len(items)
    
    # Apply source-specific filtering
    if source_type == 'exploit-db':
        filtered_items = filter_exploitdb_data(items)
    elif source_type == 'github':
        filtered_items = filter_github_data(items)
    elif source_type == 'blog':
        filtered_items = filter_blog_data(items)
    else:
        # For CTF writeups and other sources, just filter by length
        filtered_items = [item for item in items 
                         if len(item.get('content', '')) >= MIN_CONTENT_LENGTH]
    
    filtered_count = len(filtered_items)
    kept_percentage = (filtered_count / original_count * 100) if original_count > 0 else 0
    
    print(f"📄 {input_file.name}: {original_count} → {filtered_count} items ({kept_percentage:.1f}% kept)")
    
    return filtered_items if filtered_items else None

def create_consolidated_dataset(all_filtered_data):
    """Create a single consolidated training dataset"""
    output_file = OUTPUT_DIR / "consolidated_training_data.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n✅ Created consolidated dataset: {output_file.name} ({size_mb:.2f} MB)")
    
    return output_file

def main():
    print("="*80)
    print("🧹 FILTERING SCRAPED DATA FOR BOT TRAINING")
    print("="*80)
    print(f"\nInput directory: {INPUT_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}\n")
    
    all_filtered_data = []
    source_stats = Counter()
    
    # Process all JSONL files
    for input_file in sorted(INPUT_DIR.glob("*.jsonl")):
        filtered_items = process_file(input_file)
        
        if filtered_items:
            # Save individual filtered file
            output_file = OUTPUT_DIR / input_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in filtered_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # Add to consolidated dataset
            all_filtered_data.extend(filtered_items)
            source_type = filtered_items[0].get('source', 'unknown')
            source_stats[source_type] += len(filtered_items)
    
    # Create consolidated dataset
    if all_filtered_data:
        create_consolidated_dataset(all_filtered_data)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 FILTERING SUMMARY")
    print("="*80)
    print(f"\nTotal high-quality items: {len(all_filtered_data):,}")
    print("\nBreakdown by source:")
    for source, count in source_stats.most_common():
        percentage = (count / len(all_filtered_data) * 100) if all_filtered_data else 0
        print(f"  {source}: {count:,} ({percentage:.1f}%)")
    
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob('*.jsonl')) / (1024 * 1024)
    print(f"\nTotal filtered data size: {total_size:.2f} MB")
    
    print("\n" + "="*80)
    print("✅ FILTERING COMPLETE")
    print("="*80)
    print(f"\nFiltered data saved to: {OUTPUT_DIR.absolute()}")
    print("\nNext steps:")
    print("1. Review filtered_data/consolidated_training_data.jsonl")
    print("2. Use this data to train your cybersecurity bot")
    print("="*80)

if __name__ == "__main__":
    main()
