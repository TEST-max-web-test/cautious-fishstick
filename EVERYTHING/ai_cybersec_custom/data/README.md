# Cybersecurity Bot Training Data

**Last Updated**: 2025-10-21  
**Status**: ✅ Cleaned and Ready for Training

---

## 📁 Directory Structure

```
data/
├── scraped_data/                    # Original scraped data (cleaned)
│   ├── blogs.jsonl                  # 156 security blog articles
│   ├── ctf_writeups.jsonl          # 38 CTF writeups
│   ├── github_CheatSheetSeries.jsonl   # 133 OWASP cheat sheets
│   └── github_PayloadsAllTheThings.jsonl  # 167 security payloads
│
├── filtered_data/                   # ⭐ HIGH-QUALITY TRAINING DATA
│   ├── consolidated_training_data.jsonl  # 372 items - MAIN DATASET
│   ├── text_corpus.txt             # Plain text version (4.99 MB)
│   ├── blogs.jsonl                 # 142 filtered blog articles
│   ├── ctf_writeups.jsonl         # 34 filtered CTF writeups
│   ├── github_CheatSheetSeries.jsonl  # 123 filtered cheat sheets
│   ├── github_PayloadsAllTheThings.jsonl  # 71 filtered payloads
│   └── exploitdb.jsonl            # 2 filtered exploits
│
├── corpus.txt                       # Existing Q&A format (891 lines)
├── scraper.py                       # Data scraping script
├── filter_training_data.py          # Data filtering script
├── convert_to_text_corpus.py        # JSONL to text converter
├── FILTERING_SUMMARY.md             # Detailed filtering report
└── README.md                        # This file
```

---

## 🎯 Quick Start

### For Training Your Bot:

**Option 1 - Structured JSONL Format (Recommended)**
```bash
Use: filtered_data/consolidated_training_data.jsonl
Size: 5.12 MB
Items: 372 high-quality security documents
Format: One JSON object per line with metadata
```

**Option 2 - Plain Text Format**
```bash
Use: filtered_data/text_corpus.txt  
Size: 4.99 MB
Items: 372 documents in plain text
Format: Text with section headers
```

**Option 3 - Q&A Conversational Format**
```bash
Use: corpus.txt
Size: ~100 KB
Items: 891 lines of Q&A pairs
Format: User question / Agent answer
```

---

## 📊 Data Quality Metrics

### Original Scraped Data:
- **Total Items**: ~47,414
- **Total Size**: ~25 MB
- **Quality**: Mixed (contained empty files, metadata-only, configs)

### After Filtering:
- **Total Items**: 372 (99.2% reduction)
- **Total Size**: 5.12 MB (79.5% reduction)
- **Quality**: 100% security-focused, substantive content

### Retention Rates:
- Security Blogs: 91.0% kept
- OWASP Cheat Sheets: 92.5% kept
- CTF Writeups: 89.5% kept
- PayloadsAllTheThings: 42.5% kept
- ExploitDB: 0.004% kept (was 99.99% garbage)

---

## 🔍 Content Coverage

The filtered dataset covers:

### Web Application Security
- XSS, CSRF, SQL Injection
- Authentication & Authorization
- Session Management
- API Security

### Network Security
- Reconnaissance & Scanning
- Exploitation Techniques
- Post-Exploitation
- Lateral Movement

### Secure Development
- Secure Coding Practices
- C/C++ Toolchain Hardening
- Software Supply Chain Security
- Code Review Guidelines

### Penetration Testing
- CTF Techniques
- Real-world Scenarios
- Tool Usage (Burp Suite, etc.)
- Methodology

### Vulnerability Research
- CVE Analysis
- Exploit Development
- Reverse Engineering
- Malware Analysis

---

## 🛠️ Scripts & Tools

### 1. Scraper (`scraper.py`)
```bash
python3 scraper.py
```
- Scrapes security content from multiple sources
- Outputs to `scraped_data/` directory
- Sources: GitHub repos, blogs, ExploitDB, CVE databases

### 2. Filter (`filter_training_data.py`)
```bash
python3 filter_training_data.py
```
- Filters scraped data for quality
- Removes empty files, templates, configs
- Applies security keyword filtering
- Outputs to `filtered_data/` directory

### 3. Converter (`convert_to_text_corpus.py`)
```bash
python3 convert_to_text_corpus.py
```
- Converts JSONL to plain text
- Adds section headers
- Outputs to `filtered_data/text_corpus.txt`

---

## 📋 JSONL Format Structure

Each line in the JSONL files contains:

```json
{
  "source": "blog",
  "title": "WebSocket Security Testing",
  "url": "https://example.com/article",
  "content": "Full article text here...",
  "timestamp": "2025-10-19T19:19:56.511424"
}
```

Or for GitHub sources:

```json
{
  "source": "github",
  "repo": "OWASP/CheatSheetSeries",
  "file": "cheatsheets/XSS_Prevention.md",
  "url": "https://github.com/...",
  "content": "Full file content here...",
  "timestamp": "2025-10-19T19:17:21.923359"
}
```

---

## ✅ What Was Removed (Garbage)

### Empty Files (10 files):
- cve.jsonl (0 items)
- hackerone.jsonl (0 items)
- 8 empty GitHub repository files

### Low-Quality Content:
- GitHub issue templates
- Pull request templates
- Config files (.github/, .yml, .yaml)
- License files, README files
- Package manager files (package.json, requirements.txt)
- Content under 200 characters
- Non-security related content

### ExploitDB Metadata:
- 46,918 out of 46,920 items removed
- Were just file paths and descriptions
- No actual exploit code
- Kept only 2 substantive entries

---

## 🚀 Training Recommendations

### For Fine-Tuning:
1. Use `filtered_data/consolidated_training_data.jsonl`
2. Extract content field for training
3. Consider adding metadata as context

### For RAG (Retrieval Augmented Generation):
1. Use `filtered_data/consolidated_training_data.jsonl`
2. Create embeddings from content
3. Use metadata (source, title, url) for filtering

### For Conversational AI:
1. Use existing `corpus.txt` (Q&A format)
2. Or convert filtered data to Q&A format using LLM
3. Combine both datasets for comprehensive coverage

---

## 📈 Next Steps

1. ✅ **Data Collection**: Complete
2. ✅ **Data Filtering**: Complete  
3. ✅ **Data Preparation**: Complete
4. ⏭️ **Model Training**: Ready to begin
5. ⏭️ **Model Evaluation**: After training

---

## 🔄 Updating the Dataset

To refresh the dataset:

```bash
# 1. Scrape new data
python3 scraper.py

# 2. Filter the data
python3 filter_training_data.py

# 3. Convert to text (optional)
python3 convert_to_text_corpus.py
```

---

## 📝 Notes

- All empty files have been removed from `scraped_data/`
- ExploitDB file removed due to 99.99% garbage content
- Filtering is conservative to ensure high quality
- Current dataset: 372 items is optimal for fine-tuning
- Can expand by:
  - Adding more blog RSS feeds
  - Scraping more GitHub repositories
  - Including academic papers
  - Adding vulnerability databases

---

## 🆘 Support

For issues or questions:
1. Check `FILTERING_SUMMARY.md` for detailed metrics
2. Review script output for errors
3. Examine sample data in filtered files
4. Adjust filtering criteria in `filter_training_data.py`

---

**Status**: ✅ Dataset is clean, filtered, and ready for bot training!
