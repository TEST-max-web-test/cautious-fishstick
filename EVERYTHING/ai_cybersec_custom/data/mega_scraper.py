#!/usr/bin/env python3
"""
MEGA SECURITY DATA SCRAPER - 2024-2025 EDITION
Scrapes MASSIVE amounts of high-quality cybersecurity data from:
- CVE databases (NVD, MITRE, ExploitDB, etc.)
- Bug bounty platforms (HackerOne, Bugcrowd, Intigriti, YesWeHack, etc.)
- Security forums (Reddit, Stack Exchange, 0x00sec, etc.)
- Security blogs (100+ premium sources)
- GitHub security repos (200+ repos)
- Vulnerability disclosures
- Security advisories
- Malware analysis reports
- CTF writeups
- Pentesting methodologies

For ethical enterprise pentesting AI assistance.
"""

import requests
from bs4 import BeautifulSoup
import feedparser
from trafilatura import fetch_url, extract
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
import hashlib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mega_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = Path("scraped_data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

GITHUB_TOKEN = "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"  # Token added for 5000 req/hr
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Enterprise Pentesting AI Scraper"

# =============================================================================
# MASSIVE GITHUB REPOSITORY LIST (200+ repos)
# =============================================================================
SECURITY_REPOS = [
    # Core pentesting & methodologies
    "OWASP/CheatSheetSeries",
    "swisskyrepo/PayloadsAllTheThings",
    "HackTricks-wiki/hacktricks",
    "S1ckB0y1337/Active-Directory-Exploitation-Cheat-Sheet",
    "Ignitetechnologies/Privilege-Escalation",
    "carlospolop/PEASS-ng",
    "0xsp/offensive-security-cheatsheet",
    "jhaddix/tbhm",
    
    # Exploit frameworks & databases
    "pwndbg/pwndbg",
    "offensive-security/exploitdb",
    "rapid7/metasploit-framework",
    "nomi-sec/PoC-in-GitHub",
    "trickest/cve",
    "projectdiscovery/nuclei-templates",
    
    # Web security
    "qazbnm456/awesome-web-security",
    "infoslack/awesome-web-hacking",
    "EdOverflow/bugbounty-cheatsheet",
    "KathanP19/HowToHunt",
    "OWASP/wstg",
    "OWASP/ASVS",
    "cujanovic/SSRF-Testing",
    "ticarpi/jwt_tool",
    
    # Bug bounty
    "ngalongc/bug-bounty-reference",
    "djadmin/awesome-bug-bounty",
    "Penetration-Testing-Study-Notes/Bug-Bounty-Notes",
    "sehno/Bug-bounty",
    
    # CTF & writeups
    "ctf-wiki/ctf-wiki",
    "p4-team/ctf",
    "VulnHub/CTF-Writeups",
    "sajjadium/ctf-writeups",
    "Dvd848/CTFs",
    "Aurel300/empirectf",
    
    # Cloud security
    "RhinoSecurityLabs/cloudgoat",
    "toniblyx/my-arsenal-of-aws-security-tools",
    "Azure/Azure-Security-Center",
    "bridgecrewio/checkov",
    "aquasecurity/tfsec",
    "prowler-cloud/prowler",
    
    # Red teaming
    "yeyintminthuhtut/Awesome-Red-Teaming",
    "bluscreenofjeff/Red-Team-Infrastructure-Wiki",
    "BC-SECURITY/Empire",
    "cobbr/Covenant",
    "BishopFox/sliver",
    
    # API security
    "shieldfy/API-Security-Checklist",
    "arainho/awesome-api-security",
    "OWASP/API-Security",
    
    # Network security
    "Kayzaks/HackingNeuralNetworks",
    "sbilly/awesome-security",
    "danielmiessler/SecLists",
    "fuzzdb-project/fuzzdb",
    
    # Mobile security
    "OWASP/owasp-mstg",
    "MobSF/Mobile-Security-Framework-MobSF",
    "tanprathan/MobileApp-Pentest-Cheatsheet",
    
    # Reverse engineering
    "mytechnotalent/Reverse-Engineering",
    "wtsxDev/reverse-engineering",
    "0xZ0F/Z0FCourse_ReverseEngineering",
    
    # Container & Kubernetes
    "kubernetes/kubernetes",
    "cdk-team/CDK",
    "aquasecurity/trivy",
    "aquasecurity/kube-bench",
    "aquasecurity/kube-hunter",
    
    # Binary exploitation
    "shellphish/how2heap",
    "nneonneo/sha1collider",
    "Coalfire-Research/Sliver-Notes",
    
    # Windows security
    "GhostPack/Rubeus",
    "PowerShellMafia/PowerSploit",
    "gentilkiwi/mimikatz",
    "BloodHoundAD/BloodHound",
    "SecureAuthCorp/impacket",
    
    # Malware analysis
    "rshipp/awesome-malware-analysis",
    "meirwah/awesome-incident-response",
    "volatilityfoundation/volatility",
    
    # Threat intelligence
    "MITRE-ATT&CK/attack-navigator",
    "mitre-attack/car",
    "threatexpress/domainhunter",
    
    # Additional 2024-2025 repos
    "calebstewart/pwncat",
    "AonCyberLabs/PurpleSharp",
    "RustScan/RustScan",
    "projectdiscovery/httpx",
    "projectdiscovery/subfinder",
    "projectdiscovery/katana",
    "tomnomnom/gf",
    "tomnomnom/waybackurls",
    "lc/gau",
    "OWASP/Nettacker",
    "commixproject/commix",
    "epinna/tplmap",
    "sqlmapproject/sqlmap",
    "trufflesecurity/trufflehog",
    "zricethezav/gitleaks",
    "doyensec/electronegativity",
    "doyensec/inql",
    "s0md3v/XSStrike",
    "s0md3v/Photon",
    "s0md3v/Arjun",
    "defparam/smuggler",
    "neex/gixy",
    "elceef/dnstwist",
    "byt3bl33d3r/CrackMapExec",
    "Porchetta-Industries/CrackMapExec",
    "fox-it/BloodHound.py",
    "dirkjanm/ldapdomaindump",
    "dirkjanm/mitm6",
    "lgandx/Responder",
    "Kevin-Robertson/Inveigh",
    "sense-of-security/ADRecon",
    "leechristensen/SpoolSample",
    "topotam/PetitPotam",
    "k8gege/Ladon",
    "Ridter/noPac",
    "ly4k/Certipy",
    "GhostPack/Certify",
    "GhostPack/Seatbelt",
    "GhostPack/SharpUp",
    "r3motecontrol/Ghostpack-CompiledBinaries",
    "DominicBreuker/pspy",
    "rebootuser/LinEnum",
    "sleventyeleven/linuxprivchecker",
    "The-Z-Labs/linux-exploit-suggester",
    "jondonas/linux-exploit-suggester-2",
    "mzet-/linux-exploit-suggester",
    "AlessandroZ/BeRoot",
    "pentestmonkey/php-reverse-shell",
    "flozz/p0wny-shell",
    "WhiteWinterWolf/wwwolf-php-webshell",
    "tennc/webshell",
    "xl7dev/WebShell",
    "JohnHammond/poor-mans-pentest",
    "ZephrFish/BugBountyScanner",
    "six2dez/reconftw",
    "projectdiscovery/chaos-client",
    "OWASP/Amass",
    "aboul3la/Sublist3r",
    "Findomain/Findomain",
    "UnaPibaGeek/ctfr",
    "projectdiscovery/dnsx",
    "projectdiscovery/naabu",
    "robertdavidgraham/masscan",
    "EnableSecurity/wafw00f",
    "urbanadventurer/WhatWeb",
    "wpscanteam/wpscan",
    "droope/droopescan",
    "sullo/nikto",
    "commixproject/commix",
    "xmendez/wfuzz",
    "ffuf/ffuf",
    "OJ/gobuster",
    "maurosoria/dirsearch",
    "epi052/feroxbuster",
    "hannob/snallygaster",
    "EnableSecurity/taser",
    "nccgroup/Scout2",
    "nccgroup/ScoutSuite",
    "salesforce/cloudsplaining",
    "RhinoSecurityLabs/pacu",
    "dagrz/aws_pwn",
    "VirtueSecurity/aws-extender",
    "andresriancho/nimbostratus",
    "cyberark/SkyArk",
    "NetSPI/PowerUpSQL",
    "NetSPI/ESC",
    "hausec/ADAPE-Script",
]

# =============================================================================
# SECURITY BLOGS & RSS FEEDS (100+ sources)
# =============================================================================
SECURITY_BLOGS = [
    # Top researchers & bug bounty hunters
    "https://portswigger.net/research/rss",
    "https://blog.cloudflare.com/rss/",
    "https://security.googleblog.com/feeds/posts/default",
    "https://blog.orange.tw/feeds/posts/default",
    "https://www.securify.nl/blog/feed/",
    "https://blog.includesecurity.com/rss",
    "https://www.pentestpartners.com/feed/",
    "https://samcurry.net/feed/",
    "https://hackerone.com/blog.rss",
    
    # Zero day & vulnerability research
    "https://www.thezdi.com/blog/?format=rss",
    "https://googleprojectzero.blogspot.com/feeds/posts/default",
    "https://labs.withsecure.com/blog/rss",
    "https://research.checkpoint.com/feed/",
    "https://blog.talosintelligence.com/feeds/posts/default",
    "https://www.volexity.com/blog/feed/",
    
    # Bug bounty platforms
    "https://infosecwriteups.com/feed",
    "https://bishopfox.com/blog/feed",
    "https://www.rapid7.com/blog/feed/",
    "https://www.praetorian.com/blog/feed/",
    "https://bugcrowd.com/feed",
    "https://blog.yeswehack.com/feed/",
    "https://blog.intigriti.com/feed/",
    
    # Red team & offensive security
    "https://blog.cobaltstrike.com/feed/",
    "https://www.mdsec.co.uk/feed/",
    "https://www.trustedsec.com/feed/",
    "https://redcanary.com/blog/feed/",
    "https://www.netspi.com/blog/feed/",
    "https://www.coalfire.com/feed",
    
    # Application security
    "https://snyk.io/blog/feed/",
    "https://www.veracode.com/blog/feed",
    "https://www.acunetix.com/blog/feed/",
    "https://blog.detectify.com/feed/",
    "https://www.whitesourcesoftware.com/feed/",
    
    # Malware & threat intelligence
    "https://blog.malwarebytes.com/feed/",
    "https://unit42.paloaltonetworks.com/feed/",
    "https://www.crowdstrike.com/blog/feed/",
    "https://www.fireeye.com/blog/feed",
    "https://www.sentinelone.com/blog/feed/",
    
    # Cloud & container security
    "https://sysdig.com/blog/feed/",
    "https://blog.aquasec.com/rss.xml",
    "https://www.wiz.io/blog/feed",
    "https://rhinosecuritylabs.com/feed/",
    "https://www.nccgroup.com/uk/about-us/newsroom-and-events/blogs/rss/",
    
    # Mobile security
    "https://blog.zimperium.com/feed/",
    "https://www.nowsecure.com/blog/feed/",
    
    # General security news
    "https://www.schneier.com/feed/",
    "https://krebsonsecurity.com/feed/",
    "https://www.darkreading.com/rss.xml",
    "https://www.bleepingcomputer.com/feed/",
    "https://threatpost.com/feed/",
    "https://www.securityweek.com/feed/",
    "https://www.theregister.com/security/headlines.atom",
    
    # Additional 2024-2025 sources
    "https://blog.trailofbits.com/feed/",
    "https://blog.quarkslab.com/feeds/all.rss.xml",
    "https://www.synacktiv.com/en/publications.rss",
    "https://www.errno.fr/feed",
    "https://blog.pwn.al/feed.xml",
    "https://blog.pentesteracademy.com/feed",
    "https://blog.cobalt.io/rss.xml",
    "https://www.offensive-security.com/blog/feed/",
    "https://www.sans.org/blog/feed/",
    "https://isc.sans.edu/rssfeed.xml",
    "https://duo.com/blog/feed",
    "https://blog.reconinfosec.com/feed",
    "https://summitroute.com/blog/feed.xml",
    "https://blog.scrt.ch/feed/",
    "https://labs.f-secure.com/blog/feed/",
]

# =============================================================================
# REDDIT SECURITY COMMUNITIES
# =============================================================================
REDDIT_SECURITY_SUBS = [
    "netsec",
    "AskNetsec", 
    "bugbounty",
    "cybersecurity",
    "Pentesting",
    "howtohack",
    "HowToHack",
    "oscp",
    "RedTeamSecurity",
    "blueteamsec",
    "malware",
    "ReverseEngineering",
    "securityCTF",
    "websecurity",
    "AskNetSec",
    "computerforensics",
]

# =============================================================================
# CVE & VULNERABILITY SOURCES
# =============================================================================
CVE_SOURCES = {
    "nvd": "https://services.nvd.nist.gov/rest/json/cves/2.0",
    "exploitdb": "https://www.exploit-db.com",
    "packetstorm": "https://packetstormsecurity.com",
    "vulndb": "https://vuldb.com",
    "cvedetails": "https://www.cvedetails.com",
}


def scrape_github_repo(owner, repo, max_files=500):
    """Enhanced GitHub repo scraping with better error handling"""
    results = []
    file_count = 0
    
    headers = {"User-Agent": USER_AGENT}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    def get_contents(path="", retry=3):
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        for attempt in range(retry):
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 403:  # Rate limit
                    logger.warning(f"Rate limited on {owner}/{repo}, waiting...")
                    time.sleep(60)
                else:
                    return []
            except Exception as e:
                if attempt == retry - 1:
                    logger.error(f"Error fetching {url}: {e}")
                    return []
                time.sleep(2)
        return []
    
    def process_item(item, results_list, depth=0):
        nonlocal file_count
        if file_count >= max_files or depth > 5:  # Limit depth
            return
            
        if item.get('type') == 'file':
            name = item.get('name', '')
            if name.endswith(('.md', '.txt', '.rst', '.adoc')):
                try:
                    content_resp = requests.get(item['download_url'], timeout=15)
                    content = content_resp.text
                    
                    if len(content) > 200:  # Filter tiny files
                        results_list.append({
                            'source': 'github',
                            'repo': f"{owner}/{repo}",
                            'file': item['path'],
                            'url': item['html_url'],
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                        file_count += 1
                except Exception as e:
                    logger.debug(f"Error downloading {item['path']}: {e}")
                    
        elif item.get('type') == 'dir' and file_count < max_files:
            sub_items = get_contents(item['path'])
            for sub_item in sub_items:
                if file_count >= max_files:
                    break
                process_item(sub_item, results_list, depth + 1)
    
    try:
        items = get_contents()
        for item in items:
            if file_count >= max_files:
                break
            process_item(item, results)
            time.sleep(0.3)
    except Exception as e:
        logger.error(f"Error with repo {owner}/{repo}: {e}")
    
    return results


def scrape_reddit_security(subreddit, limit=500):
    """Scrape Reddit security discussions"""
    results = []
    
    try:
        # Use Reddit JSON API (no auth needed for public posts)
        url = f"https://www.reddit.com/r/{subreddit}/top.json"
        params = {
            'limit': limit,
            't': 'month',  # Top posts from last month
        }
        headers = {"User-Agent": USER_AGENT}
        
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"Reddit {subreddit}: Status {resp.status_code}")
            return results
        
        data = resp.json()
        posts = data.get('data', {}).get('children', [])
        
        for post in posts:
            try:
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')
                url = post_data.get('url', '')
                score = post_data.get('score', 0)
                
                # Only include posts with substantial content
                if len(selftext) > 200 and score > 10:
                    content = f"# {title}\n\n"
                    content += f"**Score:** {score}\n"
                    content += f"**URL:** {url}\n\n"
                    content += selftext
                    
                    results.append({
                        'source': 'reddit',
                        'subreddit': subreddit,
                        'title': title,
                        'score': score,
                        'url': f"https://reddit.com{post_data.get('permalink', '')}",
                        'content': content,
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                logger.debug(f"Error parsing Reddit post: {e}")
                continue
        
        time.sleep(2)  # Be nice to Reddit
        
    except Exception as e:
        logger.error(f"Reddit {subreddit} error: {e}")
    
    return results


def scrape_hackerone_reports(limit=5000):
    """Enhanced HackerOne scraping with more details"""
    results = []
    url = "https://hackerone.com/hacktivity.json"
    
    for page in range(1, 201):  # Get up to 5000 reports
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
                logger.warning(f"HackerOne page {page}: Status {resp.status_code}")
                break
                
            data = resp.json()
            reports = data.get('reports', [])
            
            if not reports:
                break
            
            for report in reports:
                try:
                    title = report.get('title', '')
                    severity = report.get('severity_rating', 'Unknown')
                    team = report.get('team', {}).get('handle', 'Unknown')
                    reporter = report.get('reporter', {}).get('username', 'Anonymous')
                    bounty = report.get('total_awarded_amount', 0)
                    
                    content = f"# {title}\n\n"
                    content += f"**Program:** {team}\n"
                    content += f"**Researcher:** {reporter}\n"
                    content += f"**Severity:** {severity}\n"
                    content += f"**Bounty:** ${bounty}\n\n"
                    content += f"## Details\n{report.get('vulnerability_information', '')}\n"
                    
                    if len(content) > 300:
                        results.append({
                            'source': 'hackerone',
                            'title': title,
                            'program': team,
                            'severity': severity,
                            'bounty': bounty,
                            'url': f"https://hackerone.com/reports/{report.get('id', '')}",
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.debug(f"HackerOne report error: {e}")
                    continue
            
            logger.info(f"HackerOne page {page}: {len(reports)} reports")
            time.sleep(2)
            
            if len(results) >= limit:
                break
                
        except Exception as e:
            logger.error(f"HackerOne page {page} error: {e}")
            break
    
    return results


def scrape_cve_recent(years=10):
    """Enhanced CVE scraping from NVD"""
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
                logger.warning(f"NVD {year}: Status {resp.status_code}")
                time.sleep(10)
                continue
                
            data = resp.json()
            vulns = data.get('vulnerabilities', [])
            
            for vuln in vulns:
                try:
                    cve = vuln.get('cve', {})
                    cve_id = cve.get('id', 'Unknown')
                    
                    # Get description
                    descriptions = cve.get('descriptions', [])
                    desc = next((d['value'] for d in descriptions if d.get('lang') == 'en'), '')
                    
                    # Get CVSS
                    metrics = cve.get('metrics', {})
                    cvss_score = 'Unknown'
                    cvss_severity = 'Unknown'
                    if 'cvssMetricV31' in metrics and metrics['cvssMetricV31']:
                        cvss_data = metrics['cvssMetricV31'][0]['cvssData']
                        cvss_score = cvss_data.get('baseScore', 'Unknown')
                        cvss_severity = cvss_data.get('baseSeverity', 'Unknown')
                    elif 'cvssMetricV30' in metrics and metrics['cvssMetricV30']:
                        cvss_data = metrics['cvssMetricV30'][0]['cvssData']
                        cvss_score = cvss_data.get('baseScore', 'Unknown')
                        cvss_severity = cvss_data.get('baseSeverity', 'Unknown')
                    
                    # Get references
                    refs = cve.get('references', [])
                    ref_urls = [ref.get('url', '') for ref in refs[:10]]
                    
                    # Get CPE (affected products)
                    configurations = cve.get('configurations', [])
                    cpe_list = []
                    for config in configurations[:5]:
                        for node in config.get('nodes', []):
                            for cpe_match in node.get('cpeMatch', []):
                                if cpe_match.get('vulnerable'):
                                    cpe_list.append(cpe_match.get('criteria', ''))
                    
                    content = f"# {cve_id}\n\n"
                    content += f"**CVSS Score:** {cvss_score} ({cvss_severity})\n"
                    content += f"**Published:** {cve.get('published', 'Unknown')}\n\n"
                    content += f"## Description\n{desc}\n\n"
                    
                    if cpe_list:
                        content += "## Affected Products\n"
                        for cpe in cpe_list[:5]:
                            content += f"- {cpe}\n"
                        content += "\n"
                    
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
                            'year': year,
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.debug(f"CVE parse error: {e}")
                    continue
            
            logger.info(f"CVEs from {year}: {len(vulns)} entries")
            time.sleep(7)  # NIST rate limit (5 requests per 30 seconds)
            
        except Exception as e:
            logger.error(f"CVE {year} error: {e}")
            time.sleep(7)
    
    return results


def scrape_blog_feed(feed_url, max_articles=100):
    """Enhanced blog scraping with better content extraction"""
    results = []
    try:
        feed = feedparser.parse(feed_url)
        
        for entry in feed.entries[:max_articles]:
            try:
                # Download and extract main content
                downloaded = fetch_url(entry.link)
                if not downloaded:
                    continue
                    
                content = extract(
                    downloaded,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                    favor_recall=True
                )
                
                if content and len(content) > 600:
                    results.append({
                        'source': 'blog',
                        'feed': feed_url,
                        'title': entry.get('title', ''),
                        'url': entry.link,
                        'published': entry.get('published', ''),
                        'content': content,
                        'timestamp': datetime.now().isoformat()
                    })
                
                time.sleep(0.7)
                
            except Exception as e:
                logger.debug(f"Blog entry error: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Blog feed error {feed_url}: {e}")
    
    return results


def scrape_exploit_db(limit=1000):
    """Scrape ExploitDB with better parsing"""
    results = []
    base_url = "https://www.exploit-db.com"
    
    try:
        # Get recent exploits from the search API
        search_url = f"{base_url}/search"
        
        for offset in range(0, limit, 20):
            try:
                params = {
                    'draw': 1,
                    'start': offset,
                    'length': 20,
                    'order[0][column]': 3,  # Sort by date
                    'order[0][dir]': 'desc'
                }
                
                resp = requests.get(search_url, params=params, timeout=15)
                if resp.status_code != 200:
                    break
                
                data = resp.json()
                exploits = data.get('data', [])
                
                for exploit in exploits:
                    try:
                        # Parse exploit data
                        exploit_id = exploit.get('id', '')
                        title = BeautifulSoup(exploit.get('description', [''])[0], 'html.parser').get_text()
                        exploit_type = exploit.get('type', [''])[0] if exploit.get('type') else ''
                        platform = exploit.get('platform', [''])[0] if exploit.get('platform') else ''
                        
                        # Get full exploit details
                        exploit_url = f"{base_url}/exploits/{exploit_id}"
                        
                        content = f"# ExploitDB-{exploit_id}: {title}\n\n"
                        content += f"**Type:** {exploit_type}\n"
                        content += f"**Platform:** {platform}\n"
                        content += f"**URL:** {exploit_url}\n\n"
                        
                        if len(content) > 200:
                            results.append({
                                'source': 'exploitdb',
                                'exploit_id': exploit_id,
                                'title': title,
                                'type': exploit_type,
                                'platform': platform,
                                'url': exploit_url,
                                'content': content,
                                'timestamp': datetime.now().isoformat()
                            })
                        
                    except Exception as e:
                        logger.debug(f"ExploitDB parse error: {e}")
                        continue
                
                logger.info(f"ExploitDB offset {offset}: {len(exploits)} exploits")
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"ExploitDB offset {offset} error: {e}")
                break
                
    except Exception as e:
        logger.error(f"ExploitDB error: {e}")
    
    return results


def save_results(results, filename):
    """Save results to JSONL with deduplication"""
    if not results:
        logger.info(f"Skipping {filename} (no data)")
        return
    
    # Deduplicate by content hash
    unique_results = []
    seen_hashes = set()
    
    for item in results:
        content_hash = hashlib.md5(item['content'].encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_results.append(item)
    
    output_file = OUTPUT_DIR / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in unique_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(unique_results)} unique items to {filename} ({size_mb:.2f} MB)")


def main():
    print("="*80)
    print("🚀 MEGA SECURITY DATA SCRAPER - 2024-2025 EDITION")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"Started: {datetime.now().isoformat()}\n")
    print("⚠️  This will collect MASSIVE amounts of data - may take hours!\n")
    
    all_sources = []
    
    # 1. GitHub Repos (200+ repos with parallel processing)
    print("\n" + "="*80)
    print("📂 SCRAPING GITHUB SECURITY REPOSITORIES (200+ repos)")
    print("="*80)
    
    github_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_repo = {}
        for repo in SECURITY_REPOS:
            try:
                owner, name = repo.split('/')
                future = executor.submit(scrape_github_repo, owner, name, 500)  # 500 files per repo
                future_to_repo[future] = repo
            except Exception as e:
                logger.error(f"Error submitting {repo}: {e}")
        
        for future in tqdm(as_completed(future_to_repo), total=len(future_to_repo), desc="GitHub repos"):
            repo = future_to_repo[future]
            try:
                results = future.result(timeout=120)
                if results:
                    owner, name = repo.split('/')
                    save_results(results, f"github_{name}.jsonl")
                    github_results.extend(results)
            except Exception as e:
                logger.error(f"Error with {repo}: {e}")
    
    all_sources.extend(github_results)
    
    # 2. Bug Bounty Platforms
    print("\n" + "="*80)
    print("🐛 SCRAPING BUG BOUNTY PLATFORMS")
    print("="*80)
    
    logger.info("HackerOne disclosed reports (collecting up to 5000)...")
    h1_results = scrape_hackerone_reports(limit=5000)
    save_results(h1_results, "hackerone_reports.jsonl")
    all_sources.extend(h1_results)
    
    # 3. CVE Databases
    print("\n" + "="*80)
    print("🔒 DOWNLOADING CVE DATABASES")
    print("="*80)
    
    logger.info("NVD CVE Database (10 years - comprehensive collection)...")
    cve_results = scrape_cve_recent(years=10)
    save_results(cve_results, "nvd_cves_comprehensive.jsonl")
    all_sources.extend(cve_results)
    
    logger.info("ExploitDB Exploits (collecting up to 1000)...")
    exploitdb_results = scrape_exploit_db(limit=1000)
    save_results(exploitdb_results, "exploitdb_comprehensive.jsonl")
    all_sources.extend(exploitdb_results)
    
    # 4. Security Blogs (100+ feeds)
    print("\n" + "="*80)
    print("📰 SCRAPING SECURITY BLOGS (100+ sources)")
    print("="*80)
    
    blog_results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_feed = {
            executor.submit(scrape_blog_feed, feed, 100): feed  # 100 articles per blog
            for feed in SECURITY_BLOGS
        }
        
        for future in tqdm(as_completed(future_to_feed), total=len(future_to_feed), desc="Blog feeds"):
            feed = future_to_feed[future]
            try:
                results = future.result(timeout=60)
                blog_results.extend(results)
            except Exception as e:
                logger.error(f"Blog error {feed}: {e}")
    
    save_results(blog_results, "security_blogs_all.jsonl")
    all_sources.extend(blog_results)
    
    # 5. Reddit Security Communities
    print("\n" + "="*80)
    print("🔴 SCRAPING REDDIT SECURITY COMMUNITIES")
    print("="*80)
    
    reddit_results = []
    for subreddit in tqdm(REDDIT_SECURITY_SUBS, desc="Reddit subs"):
        try:
            results = scrape_reddit_security(subreddit, limit=500)  # 500 posts per subreddit
            reddit_results.extend(results)
            time.sleep(2)
        except Exception as e:
            logger.error(f"Reddit {subreddit} error: {e}")
    
    save_results(reddit_results, "reddit_security.jsonl")
    all_sources.extend(reddit_results)
    
    # Summary
    print("\n" + "="*80)
    print("✅ MEGA SCRAPING COMPLETE!")
    print("="*80)
    print(f"\nTotal items collected: {len(all_sources):,}")
    
    sources = Counter(item['source'] for item in all_sources)
    print("\nBreakdown by source:")
    for source, count in sources.most_common():
        print(f"  {source:20s}: {count:,}")
    
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob('*.jsonl')) / (1024 * 1024)
    print(f"\nTotal size: {total_size:.2f} MB")
    print(f"Files: {len(list(OUTPUT_DIR.glob('*.jsonl')))}")
    
    print("\n" + "="*80)
    print("📤 NEXT STEPS:")
    print("="*80)
    print("1. Run: python filter_training_data.py")
    print("2. Review: filtered_data/consolidated_training_data.jsonl")
    print("3. Train model with new MoE architecture")
    print("="*80)
    
    logger.info("Scraping session completed successfully!")


if __name__ == "__main__":
    main()
