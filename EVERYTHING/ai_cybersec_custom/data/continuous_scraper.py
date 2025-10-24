#!/usr/bin/env python3
"""
CONTINUOUS SCRAPER - RUNS FOR WEEKS UNTIL MANUALLY STOPPED
Target: 2 billion tokens for 200M parameter model
Strategy: Loop through all sources repeatedly with increasing depth
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
from datetime import datetime
import hashlib
import sys

OUTPUT_DIR = Path("scraped_data_continuous")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

GITHUB_TOKEN = "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"
HEADERS_GITHUB = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

# Top security repos (expanded continuously)
REPOS = [
    "OWASP/CheatSheetSeries", "swisskyrepo/PayloadsAllTheThings", 
    "HackTricks-wiki/hacktricks", "carlospolop/PEASS-ng",
    "rapid7/metasploit-framework", "offensive-security/exploitdb",
    "danielmiessler/SecLists", "fuzzdb-project/fuzzdb",
    "OWASP/wstg", "OWASP/ASVS", "OWASP/owasp-mstg",
    "projectdiscovery/nuclei-templates", "sqlmapproject/sqlmap",
    "gentilkiwi/mimikatz", "BC-SECURITY/Empire",
    "Ignitetechnologies/Privilege-Escalation", "S1ckB0y1337/Active-Directory-Exploitation-Cheat-Sheet",
    "EdOverflow/bugbounty-cheatsheet", "KathanP19/HowToHunt",
    "enaqx/awesome-pentest", "Hack-with-Github/Awesome-Hacking",
    "sbilly/awesome-security", "qazbnm456/awesome-web-security",
    "ctfs/write-ups-2016", "p4-team/ctf", "Gallopsled/pwntools",
    "pwndbg/pwndbg", "NationalSecurityAgency/ghidra",
    "Azure/Azure-Security-Center", "aquasecurity/trivy",
]

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         CONTINUOUS SCRAPER FOR 200M MODEL                     ║
╠═══════════════════════════════════════════════════════════════╣
║  Target: 2 billion tokens (1-2 GB filtered)                   ║
║  Strategy: Endless loop, increasing depth each cycle          ║
║  Runtime: Days/weeks (until manually stopped)                 ║
║  Output: {OUTPUT_DIR}                                         
╚═══════════════════════════════════════════════════════════════╝

Started: {datetime.now()}
GitHub Token: Active
PID: {sys.executable} (check with ps aux | grep continuous_scraper)

VERIFY I'M RUNNING:
  ps aux | grep continuous_scraper
  tail -f scraped_data_continuous/../continuous_scraper.log
  ls -lth scraped_data_continuous/
  
Press Ctrl+C or kill the process to stop.
""")

def log(msg):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def save_data(data, filename):
    """Save with deduplication"""
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
    """Scrape GitHub repo"""
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
                 any(f['path'].endswith(ext) for ext in ['.md', '.txt', '.py', '.sh', '.c', '.go'])]
        
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
                                'timestamp': datetime.now().isoformat()
                            })
                
                time.sleep(0.15)  # Rate limit
                
            except Exception as e:
                continue
        
        return data
        
    except Exception as e:
        log(f"Error {owner_repo}: {e}")
        return []

def scrape_cve_year(year):
    """Scrape CVEs for a year"""
    try:
        url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?pubStartDate={year}-01-01T00:00:00.000&pubEndDate={year}-12-31T23:59:59.999"
        resp = requests.get(url, timeout=30)
        
        if resp.status_code == 200:
            cves = resp.json().get('vulnerabilities', [])
            data = []
            
            for cve_item in cves:
                cve = cve_item.get('cve', {})
                cve_id = cve.get('id', '')
                desc = ''
                
                for description in cve.get('descriptions', []):
                    if description.get('lang') == 'en':
                        desc = description.get('value', '')
                        break
                
                if desc and len(desc) > 50:
                    data.append({
                        'source': 'cve',
                        'cve_id': cve_id,
                        'content': f"{cve_id}: {desc}",
                        'year': year,
                        'timestamp': datetime.now().isoformat()
                    })
            
            return data
        
        return []
        
    except Exception as e:
        return []

# Main loop
cycle = 1
files_per_repo = 500  # Start with 500, increase each cycle

log("🚀 STARTING CONTINUOUS COLLECTION...")
log(f"Total repos: {len(REPOS)}")
log(f"Initial depth: {files_per_repo} files per repo")

while True:
    try:
        log(f"\n{'='*70}")
        log(f"🔄 CYCLE {cycle} - Depth: {files_per_repo} files per repo")
        log(f"{'='*70}")
        
        # GitHub repos
        for i, repo in enumerate(REPOS, 1):
            log(f"[Cycle {cycle}] [{i}/{len(REPOS)}] Scraping {repo} (up to {files_per_repo} files)...")
            data = scrape_github_repo(repo, max_files=files_per_repo)
            
            if data:
                new_count = save_data(data, f"github_{repo.replace('/', '_')}.jsonl")
                log(f"  ✅ {repo}: +{new_count} new items ({len(data)} total)")
            else:
                log(f"  ⚠️ {repo}: No data")
            
            time.sleep(2)  # Breathing room
        
        # CVEs (cycle through years)
        current_year = 2025
        years_to_scrape = list(range(current_year - 5, current_year + 1))  # Last 5 years
        
        log(f"\n📋 CVE Collection (years: {years_to_scrape})...")
        for year in years_to_scrape:
            log(f"  Scraping CVEs from {year}...")
            cve_data = scrape_cve_year(year)
            if cve_data:
                new_count = save_data(cve_data, f"cve_{year}.jsonl")
                log(f"  ✅ {year}: +{new_count} new CVEs")
            time.sleep(7)  # NVD rate limit
        
        # Calculate progress
        total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.jsonl"))
        total_size_mb = total_size / (1024 * 1024)
        
        log(f"\n📊 CYCLE {cycle} COMPLETE")
        log(f"   Total collected: {total_size_mb:.2f} MB")
        log(f"   Estimated tokens: {int(total_size_mb * 150000):,}")
        log(f"   Next cycle depth: {files_per_repo + 100}")
        
        # Increase depth each cycle
        cycle += 1
        files_per_repo += 100  # Get more files each time
        
        log(f"\n⏳ Sleeping 60 seconds before next cycle...\n")
        time.sleep(60)
        
    except KeyboardInterrupt:
        log("\n⚠️ STOPPED BY USER (Ctrl+C)")
        log(f"Total cycles completed: {cycle - 1}")
        break
    except Exception as e:
        log(f"❌ Error in cycle {cycle}: {e}")
        log("⏳ Sleeping 300 seconds before retry...")
        time.sleep(300)
        continue

log(f"\n✅ Scraper stopped after {cycle - 1} cycles")
log(f"Data location: {OUTPUT_DIR}")
