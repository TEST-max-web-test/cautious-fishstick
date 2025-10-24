#!/usr/bin/env python3
"""
ADVANCED DATA FILTERING & CONSOLIDATION
Filters scraped data to keep only high-quality technical content
Removes duplicates, low-quality content, and consolidates into training format
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import hashlib
from tqdm import tqdm
import unicodedata

# Configuration
SCRAPED_DIR = Path("scraped_data")
FILTERED_DIR = Path("filtered_data")
FILTERED_DIR.mkdir(exist_ok=True, parents=True)

# Quality thresholds
MIN_LENGTH = 500  # Minimum content length
MAX_LENGTH = 50000  # Maximum content length
MIN_TECHNICAL_SCORE = 0.3  # Minimum technical relevance score

# Technical keywords for scoring
TECHNICAL_KEYWORDS = {
    # Vulnerability types
    'xss', 'csrf', 'ssrf', 'sqli', 'rce', 'lfi', 'rfi', 'xxe', 'idor',
    'deserialization', 'buffer overflow', 'race condition', 'privilege escalation',
    'command injection', 'path traversal', 'directory traversal',
    
    # Tools & techniques
    'burp suite', 'metasploit', 'nmap', 'sqlmap', 'gobuster', 'ffuf',
    'nuclei', 'subfinder', 'httpx', 'amass', 'recon', 'enumeration',
    'exploit', 'payload', 'shellcode', 'reverse shell', 'bind shell',
    
    # Technologies
    'jwt', 'oauth', 'saml', 'api', 'graphql', 'websocket', 'cors',
    'csp', 'sop', 'http', 'https', 'tls', 'ssl', 'certificate',
    
    # Cloud & containers
    'aws', 'azure', 'gcp', 's3', 'ec2', 'lambda', 'kubernetes', 'docker',
    'container', 'iam', 'rbac', 'service account',
    
    # Web security
    'html', 'javascript', 'php', 'python', 'java', 'node', 'react',
    'angular', 'vue', 'sql', 'nosql', 'mongodb', 'redis', 'mysql',
    
    # Network & systems
    'tcp', 'udp', 'dns', 'smtp', 'ssh', 'rdp', 'smb', 'ldap', 'kerberos',
    'active directory', 'windows', 'linux', 'unix', 'powershell', 'bash',
    
    # Security concepts
    'authentication', 'authorization', 'session', 'cookie', 'token',
    'encryption', 'decryption', 'hash', 'salt', 'cipher', 'aes', 'rsa',
    'security', 'vulnerability', 'cve', 'exploit', 'patch', 'mitigation',
    'pentesting', 'bug bounty', 'red team', 'blue team', 'siem', 'soc',
    
    # Mobile
    'android', 'ios', 'mobile', 'apk', 'ipa', 'root', 'jailbreak',
    
    # Malware & forensics
    'malware', 'ransomware', 'trojan', 'backdoor', 'rootkit',
    'forensics', 'memory dump', 'pcap', 'packet capture',
}

# Low-quality indicators
LOW_QUALITY_PATTERNS = [
    r'subscribe to.*newsletter',
    r'follow us on',
    r'share this (post|article)',
    r'^\s*$',  # Empty lines
    r'cookie policy',
    r'privacy policy',
    r'terms of service',
    r'all rights reserved',
    r'copyright \d{4}',
]


def clean_text(text):
    """Clean and normalize text"""
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove URLs from text (keep in markdown)
    # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!?]{3,}', '!', text)
    
    # Remove control characters
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    
    return text.strip()


def calculate_technical_score(text):
    """Calculate how technical/relevant the content is"""
    text_lower = text.lower()
    
    # Count technical keywords
    keyword_matches = 0
    for keyword in TECHNICAL_KEYWORDS:
        if keyword in text_lower:
            keyword_matches += 1
    
    # Normalize by text length (keywords per 1000 chars)
    score = (keyword_matches / max(len(text), 1)) * 1000
    
    # Bonus for code blocks
    code_blocks = len(re.findall(r'```[\s\S]*?```|`[^`]+`', text))
    score += code_blocks * 0.1
    
    # Bonus for CVE mentions
    cve_mentions = len(re.findall(r'CVE-\d{4}-\d{4,}', text, re.IGNORECASE))
    score += cve_mentions * 0.2
    
    # Bonus for technical patterns
    if re.search(r'\b(function|class|def|import|include|exploit|vulnerability)\b', text_lower):
        score += 0.1
    
    return min(score, 1.0)


def is_low_quality(text):
    """Check if content is low quality"""
    text_lower = text.lower()
    
    # Check for low-quality patterns
    for pattern in LOW_QUALITY_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    
    # Check for too many repeated lines
    lines = text.split('\n')
    if len(lines) > 10:
        unique_lines = set(lines)
        if len(unique_lines) / len(lines) < 0.3:  # Less than 30% unique lines
            return True
    
    # Check for minimum word count
    words = text.split()
    if len(words) < 50:
        return True
    
    # Check for excessive short words (likely navigation/UI text)
    short_words = sum(1 for w in words if len(w) <= 2)
    if short_words / len(words) > 0.5:
        return True
    
    return False


def content_hash(text):
    """Generate hash for deduplication"""
    # Normalize text for hashing
    normalized = re.sub(r'\s+', ' ', text.lower()).strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def filter_file(file_path):
    """Filter a single JSONL file"""
    filtered_items = []
    seen_hashes = set()
    
    total_items = 0
    filtered_count = 0
    duplicate_count = 0
    low_quality_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_items += 1
                try:
                    item = json.loads(line.strip())
                    content = item.get('content', '')
                    
                    # Skip if empty
                    if not content:
                        filtered_count += 1
                        continue
                    
                    # Clean text
                    content = clean_text(content)
                    
                    # Check length
                    if len(content) < MIN_LENGTH or len(content) > MAX_LENGTH:
                        filtered_count += 1
                        continue
                    
                    # Check quality
                    if is_low_quality(content):
                        low_quality_count += 1
                        continue
                    
                    # Check technical score
                    tech_score = calculate_technical_score(content)
                    if tech_score < MIN_TECHNICAL_SCORE:
                        filtered_count += 1
                        continue
                    
                    # Check for duplicates
                    hash_val = content_hash(content)
                    if hash_val in seen_hashes:
                        duplicate_count += 1
                        continue
                    seen_hashes.add(hash_val)
                    
                    # Add technical score to item
                    item['content'] = content
                    item['technical_score'] = tech_score
                    filtered_items.append(item)
                    
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing item: {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    print(f"  {file_path.name}:")
    print(f"    Total: {total_items}, Kept: {len(filtered_items)}")
    print(f"    Filtered: {filtered_count}, Low quality: {low_quality_count}, Duplicates: {duplicate_count}")
    
    return filtered_items


def consolidate_all(output_file="consolidated_training_data.jsonl"):
    """Consolidate all filtered data into one file with global deduplication"""
    all_items = []
    seen_hashes = set()
    
    print("\n" + "="*80)
    print("🔍 FILTERING ALL SCRAPED DATA")
    print("="*80)
    
    # Get all JSONL files
    jsonl_files = list(SCRAPED_DIR.glob('*.jsonl'))
    
    if not jsonl_files:
        print("❌ No JSONL files found in scraped_data/")
        return
    
    print(f"\nFound {len(jsonl_files)} files to process\n")
    
    # Filter each file
    for file_path in tqdm(jsonl_files, desc="Processing files"):
        filtered = filter_file(file_path)
        
        # Add to consolidated list with global deduplication
        for item in filtered:
            hash_val = content_hash(item['content'])
            if hash_val not in seen_hashes:
                seen_hashes.add(hash_val)
                all_items.append(item)
    
    # Sort by technical score (highest first)
    all_items.sort(key=lambda x: x.get('technical_score', 0), reverse=True)
    
    # Save consolidated file
    output_path = FILTERED_DIR / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Create text corpus for training
    corpus_path = FILTERED_DIR / "text_corpus.txt"
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for item in all_items:
            f.write(item['content'])
            f.write('\n\n')
    
    # Statistics
    print("\n" + "="*80)
    print("✅ FILTERING COMPLETE")
    print("="*80)
    print(f"\nTotal unique items: {len(all_items):,}")
    
    # Source breakdown
    sources = defaultdict(int)
    for item in all_items:
        sources[item.get('source', 'unknown')] += 1
    
    print("\nBreakdown by source:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source:20s}: {count:,}")
    
    # Technical score distribution
    high_quality = sum(1 for item in all_items if item.get('technical_score', 0) >= 0.6)
    medium_quality = sum(1 for item in all_items if 0.3 <= item.get('technical_score', 0) < 0.6)
    
    print(f"\nQuality distribution:")
    print(f"  High (≥0.6):    {high_quality:,} ({100*high_quality/len(all_items):.1f}%)")
    print(f"  Medium (0.3-0.6): {medium_quality:,} ({100*medium_quality/len(all_items):.1f}%)")
    
    # Size info
    consolidated_size = output_path.stat().st_size / (1024 * 1024)
    corpus_size = corpus_path.stat().st_size / (1024 * 1024)
    
    print(f"\nOutput files:")
    print(f"  {output_file}: {consolidated_size:.2f} MB")
    print(f"  text_corpus.txt: {corpus_size:.2f} MB")
    
    print("\n" + "="*80)
    print("📤 NEXT STEPS:")
    print("="*80)
    print("1. Review: filtered_data/consolidated_training_data.jsonl")
    print("2. Copy text_corpus.txt to: ../combined_corpus.txt (or append)")
    print("3. Train model: python train/train.py")
    print("="*80)


if __name__ == "__main__":
    consolidate_all()
