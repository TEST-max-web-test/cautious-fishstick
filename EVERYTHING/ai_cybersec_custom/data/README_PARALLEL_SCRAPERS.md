# PARALLEL SCRAPER SYSTEM FOR 200M PARAMETER MODEL

## 🎯 Overview

**Goal**: Collect 2-4 billion tokens (2-4 GB filtered data) for 200M parameter model

**Strategy**: 5 parallel scrapers running simultaneously to cut collection time from 3-4 weeks down to 1-2 weeks

**Output**: `scraped_parallel/` directory with deduplicated JSONL files

---

## 📦 What's Included

### Scraper Scripts (5 parallel processes)

1. **parallel_scraper_1.py** - OWASP & Web Security (16 repos)
2. **parallel_scraper_2.py** - Pentesting & Red Team (14 repos)
3. **parallel_scraper_3.py** - CTF & Reverse Engineering (15 repos)
4. **parallel_scraper_4.py** - Cloud Security & Vuln Tools (17 repos)
5. **parallel_scraper_5.py** - CVE Database (complete archive 1999-2025)

### Control Scripts

- **start_all_scrapers.sh** - Launch all 5 scrapers
- **stop_all_scrapers.sh** - Stop all scrapers gracefully
- **check_progress.sh** - Real-time progress monitoring

---

## 🚀 Quick Start

### 1. Start All Scrapers

```bash
cd /workspace/EVERYTHING/ai_cybersec_custom/data
./start_all_scrapers.sh
```

**You'll see:**
```
╔══════════════════════════════════════════════════════════════════════╗
║           PARALLEL SCRAPER SYSTEM FOR 200M MODEL                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  5 Scrapers Running Simultaneously                                   ║
║  Target: 2-4 billion tokens (2-4 GB filtered)                        ║
║  Timeline: 1-2 weeks with parallel collection                        ║
║  Output: scraped_parallel/                                           ║
╚══════════════════════════════════════════════════════════════════════╝

🚀 LAUNCHING 5 PARALLEL SCRAPERS...

[1/5] Starting OWASP & Web Security scraper...
  ✅ PID: 12345
[2/5] Starting Pentesting & Red Team scraper...
  ✅ PID: 12346
[3/5] Starting CTF & Reverse Engineering scraper...
  ✅ PID: 12347
[4/5] Starting Cloud Security & Vuln Tools scraper...
  ✅ PID: 12348
[5/5] Starting CVE Database scraper...
  ✅ PID: 12349

✅ ALL 5 SCRAPERS STARTED SUCCESSFULLY
```

### 2. Check Progress (Run Anytime)

```bash
./check_progress.sh
```

**You'll see:**
```
╔══════════════════════════════════════════════════════════════════════╗
║              PARALLEL SCRAPER PROGRESS REPORT                        ║
╚══════════════════════════════════════════════════════════════════════╝

🔍 Scraper Status:
  Scraper 1: ✅ RUNNING (PID: 12345)
  Scraper 2: ✅ RUNNING (PID: 12346)
  Scraper 3: ✅ RUNNING (PID: 12347)
  Scraper 4: ✅ RUNNING (PID: 12348)
  Scraper 5: ✅ RUNNING (PID: 12349)

📊 Data Collection:
  Total Size: 125M
  Files: 45
  Total Items: 8,543
  Est. Tokens: 4,271,500
  Progress to 2B: 0.21%

📋 Per-Scraper Breakdown:
  Scraper 1: 28 MB (12 files)
  Scraper 2: 32 MB (8 files)
  Scraper 3: 19 MB (10 files)
  Scraper 4: 24 MB (9 files)
  Scraper 5: 22 MB (6 files)
```

### 3. Monitor Individual Scrapers (Optional)

```bash
# Watch scraper 1 in real-time
tail -f logs/scraper_1.log

# Watch all scrapers (in separate terminals)
tail -f logs/scraper_2.log
tail -f logs/scraper_3.log
tail -f logs/scraper_4.log
tail -f logs/scraper_5.log
```

### 4. Stop All Scrapers

```bash
./stop_all_scrapers.sh
```

---

## 📊 Expected Performance

### Collection Rate
- **Single scraper**: ~2-4 MB/hour
- **5 parallel scrapers**: ~10-20 MB/hour (5x speedup)

### Timeline Milestones

| Time | Raw Data | Filtered Data | Tokens | Viable Model |
|------|----------|---------------|--------|--------------|
| **6 hours** | 60-120 MB | 6-12 MB | 1-2M | 100-200K params |
| **1 day** | 240-480 MB | 24-48 MB | 5-10M | 500K-1M params |
| **3 days** | 720MB-1.4GB | 72-140 MB | 15-30M | 1.5-3M params |
| **1 week** | 1.7-3.4 GB | 170-340 MB | 35-70M | 3.5-7M params |
| **2 weeks** | 3.4-6.7 GB | 340-670 MB | 70-140M | 7-14M params |
| **1 month** | 7-13 GB | 700MB-1.3GB | 140-270M | 14-27M params |

**For 200M parameters (2B tokens at 10:1 ratio):**
- **Minimum (5:1 ratio, risky)**: ~2 weeks
- **Good (10:1 ratio)**: ~3-4 weeks ✅ RECOMMENDED
- **Perfect (20:1 ratio)**: ~6-8 weeks

---

## 🔍 Verification Commands

### Check if scrapers are running
```bash
ps aux | grep parallel_scraper | grep -v grep
```

### Check PIDs
```bash
cat logs/scraper_*.pid
```

### Check total data collected
```bash
du -sh scraped_parallel/
```

### Count total items
```bash
wc -l scraped_parallel/*.jsonl
```

### See most recent files
```bash
ls -lt scraped_parallel/*.jsonl | head -10
```

---

## 🛠️ Troubleshooting

### Scraper not running?

**Check logs:**
```bash
tail -50 logs/scraper_1.log  # Replace 1 with scraper number
```

**Restart specific scraper:**
```bash
# Stop it
kill $(cat logs/scraper_1.pid)

# Start it manually
nohup python3 -u parallel_scraper_1.py > logs/scraper_1.log 2>&1 &
echo $! > logs/scraper_1.pid
```

### No data being collected?

**Check GitHub API rate limit:**
- With token: 5000 requests/hour
- Without token: 60 requests/hour

**Verify token in scraper files** (should already be set):
```python
GITHUB_TOKEN = "github_pat_11BYKPBMI0u0oRVBj8YFX8..."
```

### Scrapers keep crashing?

**Common causes:**
1. Network issues (temporary, will auto-retry)
2. API rate limits (already handled with delays)
3. Out of memory (unlikely, but check with `htop`)

**Check system resources:**
```bash
htop  # or top
```

---

## 📂 Output Structure

```
scraped_parallel/
├── s1_OWASP_CheatSheetSeries.jsonl        (Scraper 1)
├── s1_swisskyrepo_PayloadsAllTheThings.jsonl
├── s2_rapid7_metasploit-framework.jsonl   (Scraper 2)
├── s2_carlospolop_PEASS-ng.jsonl
├── s3_Gallopsled_pwntools.jsonl           (Scraper 3)
├── s3_pwndbg_pwndbg.jsonl
├── s4_Azure_Azure-Security-Center.jsonl   (Scraper 4)
├── s4_aquasecurity_trivy.jsonl
├── s5_cve_2024.jsonl                      (Scraper 5)
├── s5_cve_2023.jsonl
└── ... (100+ files)

logs/
├── scraper_1.log
├── scraper_1.pid
├── scraper_2.log
├── scraper_2.pid
├── scraper_3.log
├── scraper_3.pid
├── scraper_4.log
├── scraper_4.pid
├── scraper_5.log
└── scraper_5.pid
```

---

## 🎯 What Each Scraper Collects

### Scraper 1: OWASP & Web Security
- OWASP CheatSheets
- OWASP Testing Guides
- PayloadsAllTheThings
- Bug bounty resources
- Web hacking techniques

### Scraper 2: Pentesting & Red Team
- Metasploit framework
- Exploit-DB
- Privilege escalation
- Post-exploitation tools
- Active Directory attacks

### Scraper 3: CTF & Reverse Engineering
- Pwntools, pwndbg, GEF
- Ghidra, Radare2
- CTF writeups (2014-2016)
- Binary exploitation
- Reverse engineering

### Scraper 4: Cloud Security & Vulnerability Tools
- Azure/AWS security
- Container security (Trivy, Checkov)
- Nuclei templates
- Web scanners (SQLMap, Nikto)
- Cloud penetration testing

### Scraper 5: CVE Database
- Complete CVE archive (1999-2025)
- ~100,000+ CVE entries
- CVSS scores
- Vulnerability descriptions
- References and exploits

---

## ⚙️ Advanced Configuration

### Adjust Collection Speed

Edit each `parallel_scraper_X.py` and modify:

```python
files_per_repo = 600  # Increase for more files per cycle
time.sleep(0.12)      # Decrease for faster (but risk rate limiting)
```

### Add More Repos

Edit `parallel_scraper_X.py` and add to `REPOS` list:

```python
REPOS = [
    "owner/repo1",
    "owner/repo2",
    "your/new-repo",  # Add here
]
```

### Change Output Directory

Edit each scraper:

```python
OUTPUT_DIR = Path("scraped_parallel")  # Change path
```

---

## 📈 Next Steps After Collection

### 1. Filter Data (when you have 500MB+)
```bash
python3 filter_and_consolidate.py
```

### 2. Train Tokenizer (when you have 1GB+)
```bash
cd ../tokenizer
python3 train_tokenizer.py
```

### 3. Train Model (when you have 2GB+)
```bash
cd ../train
python3 train_200m.py
```

---

## 💡 Tips

1. **Let it run**: Don't stop scrapers prematurely, they get better data with deeper cycles
2. **Monitor daily**: Run `check_progress.sh` once per day to track progress
3. **Check quality**: Sample some JSONL files to verify data quality
4. **Be patient**: Quality data takes time - 2 weeks is fast for 2B tokens!
5. **VM uptime**: Make sure your VM/codespace stays on

---

## 🆘 Support

If scrapers aren't working:

1. Run `check_progress.sh` - see what's actually running
2. Check logs: `tail -100 logs/scraper_*.log`
3. Verify processes: `ps aux | grep parallel_scraper`
4. Check data: `ls -lh scraped_parallel/`
5. Restart: `./stop_all_scrapers.sh && ./start_all_scrapers.sh`

---

## ✅ Summary

**To start collecting for 200M model:**

```bash
cd /workspace/EVERYTHING/ai_cybersec_custom/data
./start_all_scrapers.sh
```

**Check progress anytime:**
```bash
./check_progress.sh
```

**Stop when done:**
```bash
./stop_all_scrapers.sh
```

**Timeline**: 1-2 weeks → 1-2 GB → 200M parameter model ✅

---

*Last updated: October 24, 2025*
*Target: 200M parameters | 2-4 billion tokens | 1-2 weeks collection*
