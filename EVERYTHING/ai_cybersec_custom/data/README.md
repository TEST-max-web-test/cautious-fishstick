# Cybersecurity AI Training Dataset

## Overview
This dataset contains high-quality cybersecurity content scraped from 24+ GitHub repositories, totaling **44.42 MB** of filtered, security-focused training data.

## Dataset Statistics
- **Total Items**: 619 high-quality documents
- **Total Size**: 44.42 MB (consolidated)
- **Sources**: GitHub security repositories
- **Quality**: Professionally filtered, 100% security-focused

## Data Sources

### GitHub Repositories (24)
The dataset includes content from the following repositories:

#### Pentesting & Methodologies
- OWASP CheatSheetSeries (93 docs)
- PayloadsAllTheThings (41 docs)
- HackTricks (80 docs)
- Bug Hunter's Methodology (11 docs)
- HowToHunt (68 docs)
- Privilege Escalation guides (21 docs)

#### CTF & Writeups
- CTF Wiki (76 docs)
- CTF Writeups (91 docs)
- Various CTF repos (1+ docs)

#### Cloud Security
- AWS Security Tools (1 doc)
- Azure Security Center (2 docs)
- Kubernetes (35 docs)

#### Mobile & Application Security
- OWASP MSTG (69 docs)
- Mobile Security Framework (3 docs)
- Web Security repos (3 docs)

#### Exploit & Tools
- Metasploit Framework (3 docs)
- PWNdbg (5 docs)
- PEASS-ng (1 doc)
- Bug Bounty Cheatsheet (15 docs)

## File Structure

```
data/
├── scraped_data/           # Raw scraped data (56 MB)
│   ├── github_*.jsonl      # Individual repo data
│
├── filtered_data/          # Filtered high-quality data
│   ├── consolidated_training_data.jsonl  # Main dataset (44.42 MB)
│   └── github_*.jsonl      # Individual filtered files
│
├── corpus.txt              # Original Q&A format corpus
├── scraper_enhanced.py     # Enhanced scraper with 57+ repos
└── filter_training_data.py # Intelligent filtering script
```

## Data Format

The consolidated dataset is in JSONL format with the following structure:

```json
{
  "source": "github",
  "repo": "owner/repo",
  "file": "path/to/file.md",
  "url": "https://github.com/...",
  "content": "Full markdown content...",
  "timestamp": "2025-10-21T..."
}
```

## Quality Filtering

The dataset has been filtered to remove:
- ❌ Empty files and repositories
- ❌ GitHub templates and configs (.yml, .yaml, etc.)
- ❌ License, README, Contributing files
- ❌ Content shorter than 200 characters
- ❌ Non-security related content

Only content with security keywords is retained:
- vulnerability, exploit, attack, penetration testing
- XSS, CSRF, SQL injection, command injection
- privilege escalation, reverse shell, RCE
- CTF writeups, bug bounty reports
- and 30+ more security-specific terms

## Retention Rates

| Repository | Kept % |
|-----------|--------|
| Privilege-Escalation | 95.5% |
| CheatSheetSeries | 93.0% |
| CTF-Writeups | 91.0% |
| Kubernetes | 85.4% |
| HowToHunt | 80.0% |
| HackTricks | 80.0% |
| Bug Bounty Cheatsheet | 78.9% |
| CTF-Wiki | 76.0% |

## Usage

### Training a Model

```python
import json

# Load the dataset
data = []
with open('filtered_data/consolidated_training_data.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Each item contains:
# - source: where it came from
# - content: the actual text content
# - metadata: repo, file, url, timestamp

# Use the content for training
training_texts = [item['content'] for item in data]
```

### Running the Scraper

To update the dataset with fresh content:

```bash
cd data
python3 scraper_enhanced.py
```

The scraper now includes 57+ repositories covering:
- CVE exploits and PoCs
- Bug bounty write-ups
- Modern web security
- Container & Kubernetes security
- Binary exploitation
- Malware analysis
- And much more!

### Filtering New Data

After scraping:

```bash
python3 filter_training_data.py
```

This will:
1. Filter out garbage and low-quality content
2. Keep only security-relevant documents
3. Create individual filtered files
4. Generate consolidated_training_data.jsonl

## Content Coverage

The dataset covers:
- ✅ Web application security (XSS, CSRF, SQLi, etc.)
- ✅ Network penetration testing
- ✅ Privilege escalation (Linux & Windows)
- ✅ CTF techniques and writeups
- ✅ Bug bounty hunting methodologies
- ✅ Cloud security (AWS, Azure, Kubernetes)
- ✅ Mobile security testing
- ✅ Reverse engineering basics
- ✅ Exploit development concepts
- ✅ Security tool usage (Burp, Metasploit, etc.)

## Recent Updates

### 2025-10-21
- ✅ Added 33 new GitHub repositories
- ✅ Enhanced scraper with CVE PoCs, bug bounty repos
- ✅ Added modern security blogs (40+ feeds)
- ✅ Collected 56 MB of raw data
- ✅ Filtered to 44.42 MB high-quality content
- ✅ 619 security-focused documents

## Next Steps

1. **Train your model** using the consolidated dataset
2. **Update regularly** by running the scraper monthly
3. **Customize filtering** by editing filter_training_data.py
4. **Add more sources** by editing SECURITY_REPOS in scraper_enhanced.py

## License

This dataset aggregates content from various open-source repositories. Please respect the original licenses of each source repository.
