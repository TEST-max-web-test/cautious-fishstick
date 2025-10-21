#!/usr/bin/env python3
"""
ENHANCED SECURITY DATA SCRAPER - MASSIVE EDITION
Scrapes TONS of high-quality pentesting, bug bounty, and CVE sources
"""

import requests
from bs4 import BeautifulSoup
import feedparser
from trafilatura import fetch_url, extract
import json
import time
from pathlib import Path
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter

# Configuration
OUTPUT_DIR = Path("scraped_data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

GITHUB_TOKEN = None  # Set this if you have one for higher rate limits

# =============================================================================
# GITHUB PENTESTING & SECURITY REPOS (MASSIVELY EXPANDED)
# =============================================================================
SECURITY_REPOS = [
    # Original repos
    "OWASP/CheatSheetSeries",
    "swisskyrepo/PayloadsAllTheThings",
    
    # Pentesting frameworks and methodologies
    "HackTricks-wiki/hacktricks",
    "S1ckB0y1337/Active-Directory-Exploitation-Cheat-Sheet",
    "Ignitetechnologies/Privilege-Escalation",
    "carlospolop/PEASS-ng",  # Privilege escalation awesome scripts
    "0xsp/offensive-security-cheatsheet",
    "jhaddix/tbhm",  # The Bug Hunter's Methodology
    
    # Exploit collections
    "pwndbg/pwndbg",
    "offensive-security/exploitdb",
    "rapid7/metasploit-framework",  # Will scrape docs only
    
    # Web security
    "qazbnm456/awesome-web-security",
    "infoslack/awesome-web-hacking",
    "EdOverflow/bugbounty-cheatsheet",
    "KathanP19/HowToHunt",  # Bug bounty methodologies
    
    # CTF and writeups
    "ctf-wiki/ctf-wiki",
    "p4-team/ctf",
    "VulnHub/CTF-Writeups",
    
    # Cloud security
    "RhinoSecurityLabs/cloudgoat",
    "toniblyx/my-arsenal-of-aws-security-tools",
    "Azure/Azure-Security-Center",
    
    # Red teaming
    "yeyintminthuhtut/Awesome-Red-Teaming",
    "bluscreenofjeff/Red-Team-Infrastructure-Wiki",
    
    # API security
    "shieldfy/API-Security-Checklist",
    "arainho/awesome-api-security",
    
    # Network security
    "Kayzaks/HackingNeuralNetworks",
    "sbilly/awesome-security",
    
    # NEW - CVE Exploits and PoCs (2023-2025)
    "nomi-sec/PoC-in-GitHub",  # Massive collection of CVE PoCs
    "trickest/cve",  # Up-to-date CVE PoCs
    "projectdiscovery/nuclei-templates",  # Vulnerability templates
    "fkie-cad/nvd-json-data-feeds",  # NVD JSON feeds
    
    # NEW - Bug Bounty & Security Research
    "ngalongc/bug-bounty-reference",  # Bug bounty write-ups
    "djadmin/awesome-bug-bounty",  # Bug bounty resources
    "Penetration-Testing-Study-Notes/Bug-Bounty-Notes",  # Bug bounty notes
    "sehno/Bug-bounty",  # Bug bounty write-ups and tools
    
    # NEW - Modern Web Security
    "OWASP/wstg",  # Web Security Testing Guide
    "OWASP/ASVS",  # Application Security Verification Standard
    "cujanovic/SSRF-Testing",  # SSRF exploitation techniques
    "ticarpi/jwt_tool",  # JWT security testing
    
    # NEW - Mobile Security
    "OWASP/owasp-mstg",  # Mobile Security Testing Guide
    "MobSF/Mobile-Security-Framework-MobSF",  # Mobile security framework
    
    # NEW - Reverse Engineering
    "mytechnotalent/Reverse-Engineering",  # Reverse engineering tutorials
    "wtsxDev/reverse-engineering",  # Reverse engineering resources
    
    # NEW - Container & Kubernetes Security
    "kubernetes/kubernetes",  # Official K8s (docs only)
    "cdk-team/CDK",  # Container penetration toolkit
    "aquasecurity/trivy",  # Container security scanner
    
    # NEW - Binary Exploitation
    "shellphish/how2heap",  # Heap exploitation techniques
    "nneonneo/sha1collider",  # Cryptographic attacks
    "Coalfire-Research/Sliver-Notes",  # C2 framework notes
    
    # NEW - Recent Security Tools
    "calebstewart/pwncat",  # Post-exploitation platform
    "AonCyberLabs/PurpleSharp",  # Purple team tooling
    "RustScan/RustScan",  # Modern port scanner
    
    # NEW - Windows Security
    "GhostPack/Rubeus",  # Kerberos abuse
    "PowerShellMafia/PowerSploit",  # PowerShell exploitation
    "BC-SECURITY/Empire",  # Post-exploitation framework
    
    # NEW - Malware Analysis
    "rshipp/awesome-malware-analysis",  # Malware analysis resources
    "meirwah/awesome-incident-response",  # Incident response
]

def scrape_github_repo(owner, repo, max_files=200):
    """Scrape markdown/text files from GitHub repo"""
    results = []
    file_count = 0
    
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    def get_contents(path=""):
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return []
            return resp.json()
        except Exception:
            return []
    
    def process_item(item, results_list):
        nonlocal file_count
        if file_count >= max_files:
            return
            
        if item.get('type') == 'file' and item.get('name', '').endswith(('.md', '.txt')):
            try:
                content = requests.get(item['download_url'], timeout=10).text
                if len(content) > 200:  # Filter very small files
                    results_list.append({
                        'source': 'github',
                        'repo': f"{owner}/{repo}",
                        'file': item['path'],
                        'url': item['html_url'],
                        'content': content,
                        'timestamp': datetime.now().isoformat()
                    })
                    file_count += 1
            except Exception:
                pass
        elif item.get('type') == 'dir' and file_count < max_files:
            for sub_item in get_contents(item['path']):
                if file_count >= max_files:
                    break
                process_item(sub_item, results_list)
    
    for item in get_contents():
        if file_count >= max_files:
            break
        process_item(item, results)
        time.sleep(0.3)  # Rate limiting
    
    return results

# =============================================================================
# BUG BOUNTY PLATFORMS
# =============================================================================

def scrape_hackerone_reports(limit=500):
    """Scrape disclosed HackerOne reports with full details"""
    results = []
    url = "https://hackerone.com/hacktivity.json"
    
    for page in range(1, 21):  # Get ~500 reports
        try:
            params = {
                'querystring': 'disclosed:true',
                'page': page,
                'range': 'forever',
                'sort_type': 'latest_disclosable_activity_at',
                'filter': 'type:all',
                'limit': 25
            }
            
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                break
                
            data = resp.json()
            
            for report in data.get('reports', []):
                try:
                    title = report.get('title', '')
                    substate = report.get('substate', '')
                    severity = report.get('severity_rating', 'Unknown')
                    disclosed_at = report.get('disclosed_at', '')
                    reporter = report.get('reporter', {}).get('username', 'Anonymous')
                    team = report.get('team', {}).get('handle', 'Unknown')
                    
                    # Build content
                    content = f"# {title}\n\n"
                    content += f"**Program:** {team}\n"
                    content += f"**Researcher:** {reporter}\n"
                    content += f"**Severity:** {severity}\n"
                    content += f"**Status:** {substate}\n"
                    content += f"**Disclosed:** {disclosed_at}\n\n"
                    content += f"## Summary\n{report.get('vulnerability_information', '')}\n\n"
                    
                    if len(content) > 300:
                        results.append({
                            'source': 'hackerone',
                            'title': title,
                            'program': team,
                            'severity': severity,
                            'url': f"https://hackerone.com/reports/{report.get('id', '')}",
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception:
                    continue
            
            print(f"  ✓ HackerOne page {page}: {len(data.get('reports', []))} reports")
            time.sleep(2)
            
            if len(results) >= limit:
                break
                
        except Exception as e:
            print(f"  HackerOne error page {page}: {e}")
            break
    
    return results

def scrape_bugcrowd_disclosures():
    """Scrape Bugcrowd public disclosures"""
    results = []
    # Note: Bugcrowd doesn't have easy public API, would need scraping
    # Placeholder for now
    print("  ℹ️  Bugcrowd API not publicly available, skipping")
    return results

def scrape_pentesterlab_writeups():
    """Scrape high-quality pentesting write-ups from multiple sources"""
    results = []
    
    # High-quality write-up sources
    writeup_urls = [
        # Medium security publications
        "https://infosecwriteups.com/",
        "https://pentesttools.net/",
        
        # Bug bounty platforms  
        "https://hackerone.com/hacktivity",
        
        # CTF platforms
        "https://ctftime.org/writeups",
    ]
    
    for url in writeup_urls:
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Extract article links (this is simplified, each site needs custom parsing)
            articles = soup.find_all('a', href=True, limit=20)
            
            for article in articles:
                try:
                    if not article['href'].startswith('http'):
                        article_url = url.rstrip('/') + '/' + article['href'].lstrip('/')
                    else:
                        article_url = article['href']
                    
                    # Download and extract content
                    downloaded = fetch_url(article_url)
                    content = extract(downloaded, include_comments=False, no_fallback=True)
                    
                    if content and len(content) > 800:
                        results.append({
                            'source': 'pentest-writeup',
                            'origin': url,
                            'url': article_url,
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    time.sleep(2)
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"  Writeup source error {url}: {e}")
            continue
    
    return results

def scrape_recent_cve_pocs():
    """Scrape recent CVE Proof-of-Concepts from GitHub trending"""
    results = []
    
    # GitHub search for recent CVE PoCs
    search_queries = [
        "CVE-2024",
        "CVE-2023",
        "exploit poc",
        "vulnerability poc",
    ]
    
    print("  ℹ️  GitHub CVE PoC search requires authentication, using repos instead")
    return results

# =============================================================================
# CVE DATABASES (ENHANCED)
# =============================================================================

def scrape_exploit_db_recent(limit=100):
    """Scrape recent ExploitDB entries with full content"""
    results = []
    base_url = "https://www.exploit-db.com"
    
    try:
        # Get recent exploits page
        resp = requests.get(f"{base_url}/", timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find exploit links
        exploit_rows = soup.find_all('tr', limit=limit)
        
        for row in exploit_rows[:limit]:
            try:
                # Extract exploit ID and title
                link = row.find('a', href=re.compile(r'/exploits/\d+'))
                if not link:
                    continue
                
                exploit_id = link['href'].split('/')[-1]
                title = link.text.strip()
                exploit_url = f"{base_url}{link['href']}"
                
                # Get full exploit page
                exploit_resp = requests.get(exploit_url, timeout=10)
                exploit_soup = BeautifulSoup(exploit_resp.text, 'html.parser')
                
                # Extract exploit code/content
                code_block = exploit_soup.find('code')
                description = exploit_soup.find('div', {'class': 'exploit-description'})
                
                content = f"# ExploitDB-{exploit_id}: {title}\n\n"
                if description:
                    content += f"## Description\n{description.text.strip()}\n\n"
                if code_block:
                    content += f"## Exploit Code\n```\n{code_block.text.strip()}\n```\n"
                
                if len(content) > 500:
                    results.append({
                        'source': 'exploitdb',
                        'exploit_id': exploit_id,
                        'title': title,
                        'url': exploit_url,
                        'content': content,
                        'timestamp': datetime.now().isoformat()
                    })
                
                time.sleep(1)
            except Exception:
                continue
                
    except Exception as e:
        print(f"  ExploitDB error: {e}")
    
    return results

def scrape_cve_recent(years=3):
    """Download recent CVEs with detailed descriptions"""
    results = []
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    current_year = datetime.now().year
    
    for year in range(current_year - years, current_year + 1):
        try:
            params = {
                'pubStartDate': f'{year}-01-01T00:00:00.000',
                'pubEndDate': f'{year}-12-31T23:59:59.999',
                'resultsPerPage': 2000
            }
            
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"  CVE error {year}: Status {resp.status_code}")
                time.sleep(6)
                continue
                
            data = resp.json()
            
            for vuln in data.get('vulnerabilities', []):
                try:
                    cve = vuln.get('cve', {})
                    cve_id = cve.get('id', 'Unknown')
                    
                    # Get English description
                    descriptions = cve.get('descriptions', [])
                    desc = next((d['value'] for d in descriptions if d.get('lang') == 'en'), '')
                    
                    # Get CVSS score
                    metrics = cve.get('metrics', {})
                    cvss_score = 'Unknown'
                    cvss_severity = 'Unknown'
                    if 'cvssMetricV31' in metrics and metrics['cvssMetricV31']:
                        cvss_data = metrics['cvssMetricV31'][0]['cvssData']
                        cvss_score = cvss_data.get('baseScore', 'Unknown')
                        cvss_severity = cvss_data.get('baseSeverity', 'Unknown')
                    
                    # Get references
                    refs = cve.get('references', [])
                    ref_urls = [ref.get('url', '') for ref in refs[:5]]
                    
                    # Build content
                    content = f"# {cve_id}\n\n"
                    content += f"**CVSS Score:** {cvss_score} ({cvss_severity})\n"
                    content += f"**Published:** {cve.get('published', 'Unknown')}\n\n"
                    content += f"## Description\n{desc}\n\n"
                    if ref_urls:
                        content += "## References\n"
                        for url in ref_urls:
                            content += f"- {url}\n"
                    
                    if len(desc) > 100:
                        results.append({
                            'source': 'nvd-cve',
                            'cve_id': cve_id,
                            'cvss_score': cvss_score,
                            'severity': cvss_severity,
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception:
                    continue
            
            print(f"  ✓ CVEs from {year}: {len(data.get('vulnerabilities', []))} entries")
            time.sleep(6)  # NIST rate limit
            
        except Exception as e:
            print(f"  CVE error {year}: {e}")
            time.sleep(6)
    
    return results

def scrape_cve_mitre():
    """Scrape CVE from MITRE's GitHub"""
    results = []
    # MITRE CVE list on GitHub
    owner, repo = "CVEProject", "cvelistV5"
    print("  ℹ️  MITRE CVE repo is huge, using NVD API instead")
    return results

# =============================================================================
# SECURITY BLOGS (MASSIVELY EXPANDED - 2023-2025 FOCUS)
# =============================================================================
SECURITY_BLOGS = [
    # Original
    "https://portswigger.net/research/rss",
    "https://blog.cloudflare.com/rss/",
    "https://security.googleblog.com/feeds/posts/default",
    
    # Top security researchers
    "https://blog.orange.tw/feeds/posts/default",  # Orange Tsai
    "https://www.securify.nl/blog/feed/",
    "https://blog.includesecurity.com/rss",
    
    # Bug bounty focused
    "https://www.pentestpartners.com/feed/",
    "https://samcurry.net/feed/",
    "https://hackerone.com/blog.rss",
    
    # Web security
    "https://www.acunetix.com/blog/feed/",
    "https://blog.detectify.com/feed/",
    
    # Cloud security
    "https://rhinosecuritylabs.com/feed/",
    "https://www.nccgroup.com/uk/about-us/newsroom-and-events/blogs/rss/",
    
    # General security
    "https://www.schneier.com/feed/",
    "https://krebsonsecurity.com/feed/",
    "https://www.darkreading.com/rss.xml",
    "https://www.bleepingcomputer.com/feed/",
    "https://threatpost.com/feed/",
    
    # Research heavy
    "https://research.checkpoint.com/feed/",
    "https://blog.talosintelligence.com/feeds/posts/default",
    
    # NEW - Zero Day Initiative & Top Researchers
    "https://www.thezdi.com/blog/?format=rss",  # Zero Day Initiative
    "https://googleprojectzero.blogspot.com/feeds/posts/default",  # Project Zero
    "https://labs.withsecure.com/blog/rss",  # WithSecure Labs
    
    # NEW - Bug Bounty Hunters & Security Researchers
    "https://infosecwriteups.com/feed",  # InfoSec Write-ups
    "https://bishopfox.com/blog/feed",  # Bishop Fox
    "https://www.rapid7.com/blog/feed/",  # Rapid7
    "https://www.praetorian.com/blog/feed/",  # Praetorian
    
    # NEW - Red Team & Offensive Security
    "https://blog.cobaltstrike.com/feed/",  # Cobalt Strike
    "https://www.mdsec.co.uk/feed/",  # MDSec
    "https://www.trustedsec.com/feed/",  # TrustedSec
    "https://redcanary.com/blog/feed/",  # Red Canary
    
    # NEW - Application Security
    "https://snyk.io/blog/feed/",  # Snyk
    "https://blog.sqreen.com/feed/",  # Sqreen
    "https://www.veracode.com/blog/feed",  # Veracode
    
    # NEW - Malware & Threat Intelligence
    "https://blog.malwarebytes.com/feed/",  # Malwarebytes
    "https://unit42.paloaltonetworks.com/feed/",  # Unit 42
    "https://www.crowdstrike.com/blog/feed/",  # CrowdStrike
    
    # NEW - Cloud & Container Security
    "https://sysdig.com/blog/feed/",  # Sysdig
    "https://blog.aquasec.com/rss.xml",  # Aqua Security
    "https://www.wiz.io/blog/feed",  # Wiz
    
    # NEW - Mobile Security
    "https://blog.zimperium.com/feed/",  # Zimperium
    "https://www.nowsecure.com/blog/feed/",  # NowSecure
    
    # NEW - DFIR & Forensics
    "https://www.fireeye.com/blog/feed",  # Mandiant/FireEye
    "https://www.volexity.com/blog/feed/",  # Volexity
]

def scrape_blog_feed(feed_url, max_articles=50):
    """Scrape RSS feed and extract article content"""
    results = []
    try:
        feed = feedparser.parse(feed_url)
        
        for entry in feed.entries[:max_articles]:
            try:
                downloaded = fetch_url(entry.link)
                content = extract(downloaded,
                                include_comments=False,
                                include_tables=False,
                                no_fallback=True)
                
                if content and len(content) > 500:
                    results.append({
                        'source': 'blog',
                        'feed': feed_url,
                        'title': entry.get('title', ''),
                        'url': entry.link,
                        'published': entry.get('published', ''),
                        'content': content,
                        'timestamp': datetime.now().isoformat()
                    })
                
                time.sleep(0.5)
                
            except Exception:
                continue
        
    except Exception as e:
        print(f"  Feed error {feed_url}: {e}")
    
    return results

# =============================================================================
# GITHUB SECURITY ADVISORIES
# =============================================================================

def scrape_github_advisories(limit=500):
    """Scrape GitHub Security Advisories"""
    results = []
    
    # GitHub Advisory Database
    owner, repo = "github", "advisory-database"
    base_url = f"https://api.github.com/repos/{owner}/{repo}/contents/advisories/github-reviewed"
    
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    print("  ℹ️  GitHub advisories repo is huge, getting sample")
    # Sample from different years
    for year in ["2024", "2023", "2022"]:
        try:
            url = f"{base_url}/{year}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
                
            month_dirs = resp.json()
            for month_dir in month_dirs[:3]:  # Sample 3 months per year
                if month_dir.get('type') != 'dir':
                    continue
                    
                month_resp = requests.get(month_dir['url'], headers=headers, timeout=10)
                if month_resp.status_code != 200:
                    continue
                    
                advisories = month_resp.json()
                for advisory in advisories[:20]:  # 20 per month
                    if advisory.get('type') != 'file':
                        continue
                    
                    try:
                        content_resp = requests.get(advisory['download_url'], timeout=10)
                        advisory_json = content_resp.json()
                        
                        content = f"# {advisory_json.get('summary', 'Security Advisory')}\n\n"
                        content += f"**ID:** {advisory_json.get('id', 'Unknown')}\n"
                        content += f"**Severity:** {advisory_json.get('severity', 'Unknown')}\n"
                        content += f"**Package:** {advisory_json.get('package', {}).get('name', 'Unknown')}\n\n"
                        content += f"## Description\n{advisory_json.get('description', '')}\n"
                        
                        if len(content) > 300:
                            results.append({
                                'source': 'github-advisory',
                                'advisory_id': advisory_json.get('id', ''),
                                'severity': advisory_json.get('severity', ''),
                                'content': content,
                                'timestamp': datetime.now().isoformat()
                            })
                    except Exception:
                        continue
                
                time.sleep(1)
                
                if len(results) >= limit:
                    break
            
            if len(results) >= limit:
                break
                
        except Exception as e:
            print(f"  Advisory error {year}: {e}")
    
    return results

# =============================================================================
# PENTESTING WRITEUPS & CTF
# =============================================================================

def scrape_ctf_writeups_enhanced():
    """Enhanced CTF writeup scraping"""
    results = []
    
    # Known CTF writeup repos
    writeup_repos = [
        "sajjadium/ctf-writeups",
        "p4-team/ctf",
        "VulnHub/writeups",
        "Dvd848/CTFs",
        "Aurel300/empirectf",
    ]
    
    for repo_str in writeup_repos:
        try:
            owner, name = repo_str.split('/')
            repo_results = scrape_github_repo(owner, name, max_files=50)
            results.extend(repo_results)
            time.sleep(2)
        except Exception as e:
            print(f"  CTF repo error {repo_str}: {e}")
    
    return results

def scrape_pentesting_blogs():
    """Scrape pentesting methodology blogs"""
    urls = [
        "https://book.hacktricks.xyz/",  # HackTricks
        "https://pentestbook.six2dez.com/",  # Pentesting book
    ]
    
    results = []
    # These would require specific scrapers - placeholder
    print("  ℹ️  Pentesting methodology sites require custom scrapers")
    return results

# =============================================================================
# PACKETSTORM SECURITY
# =============================================================================

def scrape_packetstorm():
    """Scrape recent PacketStorm advisories"""
    results = []
    base_url = "https://packetstormsecurity.com"
    
    # Get recent files page
    try:
        resp = requests.get(f"{base_url}/files/", timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find file links
        file_links = soup.find_all('a', href=re.compile(r'/files/\d+/'))
        
        for link in file_links[:100]:  # Get 100 recent items
            try:
                file_url = base_url + link['href']
                file_resp = requests.get(file_url, timeout=10)
                file_soup = BeautifulSoup(file_resp.text, 'html.parser')
                
                title = file_soup.find('h1')
                description = file_soup.find('div', {'class': 'file-content'})
                
                if title and description:
                    content = f"# {title.text.strip()}\n\n"
                    content += description.text.strip()
                    
                    if len(content) > 300:
                        results.append({
                            'source': 'packetstorm',
                            'title': title.text.strip(),
                            'url': file_url,
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                
                time.sleep(1)
            except Exception:
                continue
                
    except Exception as e:
        print(f"  PacketStorm error: {e}")
    
    return results

# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def save_results(results, filename):
    """Save results to JSONL"""
    if not results:
        print(f"  ⏭️  Skipping {filename} (no data)")
        return
        
    output_file = OUTPUT_DIR / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved {len(results)} items to {filename} ({size_mb:.2f} MB)")

def main():
    print("="*80)
    print("🚀 ENHANCED SECURITY DATA SCRAPER - MASSIVE EDITION")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"Started: {datetime.now().isoformat()}\n")
    print("⚠️  This will take a while - grabbing TONS of data!\n")
    
    all_sources = []
    
    # 1. GitHub Repos (EXPANDED)
    print("\n📂 Scraping GitHub security repositories...")
    for repo in tqdm(SECURITY_REPOS, desc="GitHub repos"):
        try:
            owner, name = repo.split('/')
            results = scrape_github_repo(owner, name, max_files=100)
            all_sources.extend(results)
            if results:
                save_results(results, f"github_{name}.jsonl")
            time.sleep(1)
        except Exception as e:
            print(f"  Error with {repo}: {e}")
    
    # 2. Bug Bounty Platforms
    print("\n🐛 Scraping bug bounty platforms...")
    
    print("  📋 HackerOne disclosed reports...")
    h1_results = scrape_hackerone_reports(limit=500)
    all_sources.extend(h1_results)
    save_results(h1_results, "hackerone_reports.jsonl")
    
    # 3. CVE Databases & Exploit Collections
    print("\n🔒 Downloading CVE databases and exploits...")
    
    print("  📊 NVD Recent CVEs...")
    cve_results = scrape_cve_recent(years=3)
    all_sources.extend(cve_results)
    save_results(cve_results, "nvd_cves.jsonl")
    
    print("  💣 ExploitDB Recent Exploits...")
    exploitdb_results = scrape_exploit_db_recent(limit=100)
    all_sources.extend(exploitdb_results)
    save_results(exploitdb_results, "exploitdb_recent.jsonl")
    
    # 4. GitHub Security Advisories
    print("\n🛡️  Scraping GitHub Security Advisories...")
    advisory_results = scrape_github_advisories(limit=300)
    all_sources.extend(advisory_results)
    save_results(advisory_results, "github_advisories.jsonl")
    
    # 5. Security Blogs (EXPANDED)
    print("\n📰 Scraping security blogs...")
    blog_results = []
    for feed in tqdm(SECURITY_BLOGS, desc="Blog feeds"):
        try:
            results = scrape_blog_feed(feed, max_articles=30)
            blog_results.extend(results)
            time.sleep(1)
        except Exception as e:
            print(f"  Blog error: {e}")
    all_sources.extend(blog_results)
    save_results(blog_results, "security_blogs.jsonl")
    
    # 6. CTF Writeups (ENHANCED)
    print("\n🏴 Scraping CTF writeups...")
    ctf_results = scrape_ctf_writeups_enhanced()
    all_sources.extend(ctf_results)
    save_results(ctf_results, "ctf_writeups.jsonl")
    
    # 7. PacketStorm
    print("\n💥 Scraping PacketStorm Security...")
    ps_results = scrape_packetstorm()
    all_sources.extend(ps_results)
    save_results(ps_results, "packetstorm.jsonl")
    
    # 8. Pentesting Write-ups (NEW)
    print("\n📝 Scraping high-quality pentesting write-ups...")
    writeup_results = scrape_pentesterlab_writeups()
    all_sources.extend(writeup_results)
    save_results(writeup_results, "pentest_writeups.jsonl")
    
    # Summary
    print("\n" + "="*80)
    print("✅ SCRAPING COMPLETE")
    print("="*80)
    print(f"\nTotal items collected: {len(all_sources):,}")
    
    sources = Counter(item['source'] for item in all_sources)
    print("\nBreakdown by source:")
    for source, count in sources.most_common():
        print(f"  {source}: {count:,}")
    
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob('*.jsonl')) / (1024 * 1024)
    print(f"\nTotal size: {total_size:.2f} MB")
    
    print("\n" + "="*80)
    print("📤 NEXT STEPS:")
    print("="*80)
    print("1. Run filter_training_data.py to clean the data")
    print("2. Review filtered_data/consolidated_training_data.jsonl")
    print("3. Train your model on the cleaned data")
    print("="*80)

if __name__ == "__main__":
    main()
