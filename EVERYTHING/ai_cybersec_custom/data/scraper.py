#!/usr/bin/env python3
"""
UNIFIED SECURITY DATA SCRAPER
Scrapes all sources, extracts clean content, outputs JSONL for AI filtering
Run: python scraper.py
Then upload output files to Claude for filtering
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
# GITHUB SECURITY REPOS
# =============================================================================
SECURITY_REPOS = [
    "OWASP/CheatSheetSeries",
    "swisskyrepo/PayloadsAllTheThings",
    "danielmiessler/SecLists",
    "The-Art-of-Hacking/h4cker",
    "qazbnm456/awesome-web-security",
    "paragonie/awesome-appsec",
    "enaqx/awesome-pentest",
    "Hack-with-Github/Awesome-Hacking",
    "trimstray/the-book-of-secret-knowledge",
    "infosecn1nja/Red-Teaming-Toolkit",
]

def scrape_github_repo(owner, repo):
    """Scrape markdown/text files from GitHub repo"""
    results = []

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
        if item.get('type') == 'file' and item.get('name', '').endswith(('.md', '.txt')):
            try:
                content = requests.get(item['download_url'], timeout=10).text
                results_list.append({
                    'source': 'github',
                    'repo': f"{owner}/{repo}",
                    'file': item['path'],
                    'url': item['html_url'],
                    'content': content,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception:
                pass
        elif item.get('type') == 'dir':
            for sub_item in get_contents(item['path']):
                process_item(sub_item, results_list)

    for item in get_contents():
        process_item(item, results)
        time.sleep(0.5)  # Rate limiting

    return results

# =============================================================================
# BUG BOUNTY REPORTS (HackerOne)
# =============================================================================
def scrape_hackerone():
    """Scrape disclosed HackerOne reports"""
    results = []
    url = "https://hackerone.com/graphql"

    for page in range(1, 50):
        query = {
            "query": """query Hacktivity($querystring: String, $first: Int, $after: String) {
                hacktivity_items(querystring: $querystring, first: $first, after: $after) {
                    edges {
                        node {
                            ... on Disclosed {
                                id
                                url
                                title
                                disclosed_at
                                severity_rating
                                report {
                                    vulnerability_information
                                    summary
                                }
                            }
                        }
                    }
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                }
            }""",
            "variables": {
                "querystring": "disclosed:true",
                "first": 25,
                "after": None
            }
        }

        try:
            resp = requests.post(url, json=query, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for edge in data['data']['hacktivity_items']['edges']:
                node = edge['node']
                report = node.get('report', {})

                content = f"{node.get('title', '')}\n\n"
                content += f"Severity: {node.get('severity_rating', 'Unknown')}\n\n"
                content += report.get('vulnerability_information', '')
                content += "\n\n" + report.get('summary', '')

                if len(content) > 200:
                    results.append({
                        'source': 'hackerone',
                        'title': node.get('title'),
                        'url': node.get('url'),
                        'severity': node.get('severity_rating'),
                        'content': content,
                        'timestamp': datetime.now().isoformat()
                    })

            if not data['data']['hacktivity_items']['pageInfo']['hasNextPage']:
                break

            time.sleep(2)

        except Exception as e:
            print(f"HackerOne error page {page}: {e}")
            break

    return results

# =============================================================================
# CVE DATABASE (NVD)
# =============================================================================
def scrape_cve_database():
    """Download CVE data from NIST NVD"""
    results = []
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    for year in range(2020, 2025):
        try:
            params = {
                'pubStartDate': f'{year}-01-01T00:00:00.000',
                'pubEndDate': f'{year}-12-31T23:59:59.999',
                'resultsPerPage': 2000
            }

            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for vuln in data.get('vulnerabilities', []):
                cve = vuln.get('cve', {})
                cve_id = cve.get('id', 'Unknown')
                descriptions = cve.get('descriptions', [])
                desc = next((d['value'] for d in descriptions if d.get('lang') == 'en'), '')
                metrics = cve.get('metrics', {})
                cvss_score = 'Unknown'
                if 'cvssMetricV31' in metrics:
                    cvss_score = metrics['cvssMetricV31'][0]['cvssData'].get('baseScore', 'Unknown')

                content = f"CVE ID: {cve_id}\n"
                content += f"CVSS Score: {cvss_score}\n\n"
                content += desc

                if len(content) > 100:
                    results.append({
                        'source': 'nvd',
                        'cve_id': cve_id,
                        'cvss_score': cvss_score,
                        'content': content,
                        'timestamp': datetime.now().isoformat()
                    })

            print(f"✓ CVEs from {year}: {len(data.get('vulnerabilities', []))}")
            time.sleep(6)  # NIST rate limit

        except Exception as e:
            print(f"CVE error {year}: {e}")

    return results

# =============================================================================
# SECURITY BLOGS
# =============================================================================
SECURITY_BLOGS = [
    "https://portswigger.net/research/rss",
    "https://blog.cloudflare.com/rss/",
    "https://security.googleblog.com/feeds/posts/default",
    "https://www.microsoft.com/security/blog/feed/",
    "https://aws.amazon.com/blogs/security/feed/",
    "https://www.schneier.com/feed/",
    "https://krebsonsecurity.com/feed/",
    "https://www.darkreading.com/rss.xml",
    "https://threatpost.com/feed/",
    "https://www.bleepingcomputer.com/feed/",
]

def scrape_blog_feed(feed_url):
    """Scrape RSS feed and extract article content"""
    results = []
    try:
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:50]:
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

                time.sleep(1)

            except Exception:
                continue

    except Exception as e:
        print(f"Feed error {feed_url}: {e}")

    return results

# =============================================================================
# EXPLOIT-DB
# =============================================================================
def scrape_exploit_db():
    """Download Exploit-DB database"""
    results = []
    csv_url = "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv"

    try:
        resp = requests.get(csv_url, timeout=30)
        resp.raise_for_status()

        import csv
        from io import StringIO

        reader = csv.DictReader(StringIO(resp.text))

        for row in reader:
            content = f"Exploit: {row.get('description', '')}\n"
            content += f"Type: {row.get('type', '')}\n"
            content += f"Platform: {row.get('platform', '')}\n"
            content += f"Author: {row.get('author', '')}\n\n"
            content += f"Path: {row.get('file', '')}"

            results.append({
                'source': 'exploit-db',
                'exploit_id': row.get('id'),
                'description': row.get('description'),
                'type': row.get('type'),
                'content': content,
                'timestamp': datetime.now().isoformat()
            })

        print(f"✓ Exploit-DB entries: {len(results)}")

    except Exception as e:
        print(f"Exploit-DB error: {e}")

    return results

# =============================================================================
# CTF WRITEUPS
# =============================================================================
def scrape_ctf_writeups():
    """Find CTF writeup repositories"""
    results = []
    search_url = "https://api.github.com/search/repositories"
    params = {
        'q': 'ctf writeup security',
        'sort': 'stars',
        'order': 'desc',
        'per_page': 30
    }

    headers = {}
    if GITHUB_TOKEN:
        headers['Authorization'] = f"token {GITHUB_TOKEN}"

    try:
        resp = requests.get(search_url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        repos = resp.json().get('items', [])

        for repo in repos[:10]:
            owner = repo['owner']['login']
            name = repo['name']
            results.extend(scrape_github_repo(owner, name))

    except Exception as e:
        print(f"CTF scraping error: {e}")

    return results

# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================
def save_results(results, filename):
    """Save results to JSONL"""
    output_file = OUTPUT_DIR / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✓ Saved {len(results)} items to {filename} ({size_mb:.2f} MB)")

def main():
    print("="*80)
    print("🚀 UNIFIED SECURITY DATA SCRAPER")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"Started: {datetime.now().isoformat()}\n")

    all_sources = []

    # 1. GitHub Repos
    print("\n📂 Scraping GitHub repositories...")
    for repo in tqdm(SECURITY_REPOS):
        owner, name = repo.split('/')
        results = scrape_github_repo(owner, name)
        all_sources.extend(results)
        save_results(results, f"github_{name}.jsonl")
        time.sleep(2)

    # 2. HackerOne
    print("\n🐛 Scraping HackerOne disclosed reports...")
    h1_results = scrape_hackerone()
    all_sources.extend(h1_results)
    save_results(h1_results, "hackerone.jsonl")

    # 3. CVE Database
    print("\n🔒 Downloading CVE database...")
    cve_results = scrape_cve_database()
    all_sources.extend(cve_results)
    save_results(cve_results, "cve.jsonl")

    # 4. Security Blogs
    print("\n📰 Scraping security blogs...")
    for feed in tqdm(SECURITY_BLOGS):
        results = scrape_blog_feed(feed)
        all_sources.extend(results)
        time.sleep(2)
    save_results([r for r in all_sources if r['source'] == 'blog'], "blogs.jsonl")

    # 5. Exploit-DB
    print("\n💣 Downloading Exploit-DB...")
    exploit_results = scrape_exploit_db()
    all_sources.extend(exploit_results)
    save_results(exploit_results, "exploitdb.jsonl")

    # 6. CTF Writeups
    print("\n🏴 Scraping CTF writeups...")
    ctf_results = scrape_ctf_writeups()
    all_sources.extend(ctf_results)
    save_results(ctf_results, "ctf_writeups.jsonl")

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
    print("1. Upload files from scraped_data to your filtering pipeline")
    print("2. Ask your filter to clean and prepare data for training")
    print("3. Train your model on the cleaned data")
    print("="*80)

if __name__ == "__main__":
    # Ensure required packages are available (optional convenience auto-install)
    required_packages = [
        "requests", "beautifulsoup4", "feedparser",
        "trafilatura", "tqdm"
    ]
    try:
        import importlib
        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                print(f"Warning: package {package} not found. Install it if needed.")
    except Exception:
        pass

    main()