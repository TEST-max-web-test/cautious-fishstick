#!/usr/bin/env python3
"""
PARALLEL SCRAPER #1 - OWASP & Web Security
Part of 5-scraper parallel system for 200M parameter model
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
from datetime import datetime
import hashlib
import sys

OUTPUT_DIR = Path("scraped_parallel")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

GITHUB_TOKEN = "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"
HEADERS_GITHUB = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

# SCRAPER 1: OWASP and Web Security repos
REPOS = [
    "OWASP/CheatSheetSeries", "OWASP/wstg", "OWASP/ASVS", "OWASP/owasp-mstg",
    "OWASP/API-Security", "OWASP/Top10", "OWASP/WebGoat", "OWASP/Juice-Shop",
    "swisskyrepo/PayloadsAllTheThings", "qazbnm456/awesome-web-security",
    "infoslack/awesome-web-hacking", "EdOverflow/bugbounty-cheatsheet",
    "KathanP19/HowToHunt", "nahamsec/Resources-for-Beginner-Bug-Bounty-Hunters",
    "pentestmonkey/php-reverse-shell", "tennc/webshell",
]

def log(msg):
    print(f"[SCRAPER-1] [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def save_data(data, filename):
    if not data:
        return 0
    filepath = OUTPUT_DIR / filename
    seen = set()
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    h = hashlib.md5(item['content'].encode()).hexdigest()
                    seen.add(h)
                except:
                    pass
    new_count = 0
    with open(filepath, 'a', encoding='utf-8') as f:
        for item in data:
            h = hashlib.md5(item['content'].encode()).hexdigest()
            if h not in seen:
                f.write(json.dumps(item) + '\n')
                seen.add(h)
                new_count += 1
    return new_count

def scrape_github_repo(owner_repo, max_files=1000):
    try:
        owner, repo = owner_repo.split('/')
        data = []
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
        resp = requests.get(url, headers=HEADERS_GITHUB, timeout=15)
        if resp.status_code != 200:
            url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1"
            resp = requests.get(url, headers=HEADERS_GITHUB, timeout=15)
        if resp.status_code != 200:
            return []
        tree = resp.json().get('tree', [])
        files = [f for f in tree if f['type'] == 'blob' and 
                 any(f['path'].endswith(ext) for ext in ['.md', '.txt', '.py', '.sh', '.c', '.go', '.java'])]
        for file_info in files[:max_files]:
            try:
                content_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_info['path']}"
                resp = requests.get(content_url, headers=HEADERS_GITHUB, timeout=10)
                if resp.status_code == 200:
                    content_json = resp.json()
                    if 'content' in content_json:
                        import base64
                        content = base64.b64decode(content_json['content']).decode('utf-8', errors='ignore')
                        if len(content) > 200:
                            data.append({
                                'source': 'github',
                                'repo': owner_repo,
                                'file': file_info['path'],
                                'content': content,
                                'scraper': 1,
                                'timestamp': datetime.now().isoformat()
                            })
                time.sleep(0.12)
            except Exception:
                continue
        return data
    except Exception as e:
        return []

log("🚀 SCRAPER 1 STARTED: OWASP & Web Security")
cycle = 1
files_per_repo = 600

while True:
    try:
        log(f"🔄 CYCLE {cycle} - Depth: {files_per_repo} files/repo")
        for i, repo in enumerate(REPOS, 1):
            log(f"[{i}/{len(REPOS)}] {repo}...")
            data = scrape_github_repo(repo, max_files=files_per_repo)
            if data:
                new = save_data(data, f"s1_{repo.replace('/', '_')}.jsonl")
                log(f"  ✅ +{new} items")
            time.sleep(2)
        
        total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("s1_*.jsonl"))
        log(f"✅ CYCLE {cycle} DONE | Total: {total_size/(1024*1024):.1f} MB")
        cycle += 1
        files_per_repo += 100
        time.sleep(60)
    except KeyboardInterrupt:
        log("STOPPED")
        break
    except Exception as e:
        log(f"ERROR: {e}")
        time.sleep(300)
