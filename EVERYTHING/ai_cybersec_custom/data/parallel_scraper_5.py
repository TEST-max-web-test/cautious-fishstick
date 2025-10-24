#!/usr/bin/env python3
"""
PARALLEL SCRAPER #5 - CVE Database Continuous Collection
Part of 5-scraper parallel system for 200M parameter model
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime
import hashlib

OUTPUT_DIR = Path("scraped_parallel")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def log(msg):
    print(f"[SCRAPER-5] [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

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

def scrape_cve_year(year):
    """Scrape CVEs for a specific year"""
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
                
                # Get CVSS scores
                metrics = cve.get('metrics', {})
                cvss_v3 = metrics.get('cvssMetricV31', [{}])[0] if metrics.get('cvssMetricV31') else {}
                base_score = cvss_v3.get('cvssData', {}).get('baseScore', 'N/A')
                
                # Get references
                refs = cve.get('references', [])
                ref_urls = [r.get('url', '') for r in refs[:5]]
                
                if desc and len(desc) > 50:
                    content = f"CVE ID: {cve_id}\n"
                    content += f"Description: {desc}\n"
                    if base_score != 'N/A':
                        content += f"CVSS Score: {base_score}\n"
                    if ref_urls:
                        content += f"References: {', '.join(ref_urls)}\n"
                    
                    data.append({
                        'source': 'cve',
                        'cve_id': cve_id,
                        'content': content,
                        'year': year,
                        'scraper': 5,
                        'timestamp': datetime.now().isoformat()
                    })
            
            return data
        return []
    except Exception as e:
        log(f"Error year {year}: {e}")
        return []

log("🚀 SCRAPER 5 STARTED: CVE Database Collection")
cycle = 1

while True:
    try:
        log(f"🔄 CYCLE {cycle} - CVE Collection")
        
        # Scrape from 1999 to current year
        current_year = 2025
        years = list(range(1999, current_year + 1))
        
        for year in years:
            log(f"Scraping CVEs from {year}...")
            cve_data = scrape_cve_year(year)
            
            if cve_data:
                new = save_data(cve_data, f"s5_cve_{year}.jsonl")
                log(f"  ✅ {year}: +{new} new CVEs")
            
            time.sleep(7)  # NVD API rate limit (6 seconds + buffer)
        
        total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("s5_*.jsonl"))
        log(f"✅ CYCLE {cycle} DONE | Total: {total_size/(1024*1024):.1f} MB")
        
        cycle += 1
        log("⏳ Sleeping 300 seconds before next CVE cycle...")
        time.sleep(300)  # Wait 5 minutes between full cycles
        
    except KeyboardInterrupt:
        log("STOPPED")
        break
    except Exception as e:
        log(f"ERROR: {e}")
        time.sleep(600)
