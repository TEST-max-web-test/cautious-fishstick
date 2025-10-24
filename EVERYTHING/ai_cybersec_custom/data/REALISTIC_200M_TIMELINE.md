# REALISTIC 200M PARAMETER TIMELINE

## THE HARD TRUTH

**To train 200M parameters with ZERO underfitting:**
- Minimum: 1 billion tokens (5:1 ratio - risky)
- Good: 2 billion tokens (10:1 ratio)
- Perfect: 4 billion tokens (20:1 ratio - Chinchilla optimal)

**Current progress:**
- Have: 2 million tokens
- Need: 1,000-2,000x MORE

## REALISTIC COLLECTION TIMELINE

### Week 1 (Days 1-7)
- **Target**: 50-100M tokens
- **Filtered**: 50-100 MB
- **Sources**: GitHub deep scrapes, CVE archives, Exploit-DB
- **Status**: Foundation building

### Week 2 (Days 8-14)
- **Target**: 200-300M tokens (cumulative)
- **Filtered**: 200-300 MB
- **Sources**: Add blogs, forums, documentation
- **Status**: Scaling up

### Week 3 (Days 15-21)
- **Target**: 500M-1B tokens (cumulative)
- **Filtered**: 500MB-1GB
- **Sources**: Add mailing lists, conference talks, papers
- **Status**: First viable model (5:1 ratio, 200M params)

### Week 4 (Days 22-30)
- **Target**: 1-2B tokens (cumulative)
- **Filtered**: 1-2 GB
- **Sources**: Complete all source types
- **Status**: Good model (10:1 ratio, 200M params) ✅

### Month 2 (Optional)
- **Target**: 2-4B tokens (cumulative)
- **Filtered**: 2-4 GB
- **Sources**: Expand existing, add new sources
- **Status**: Perfect model (20:1 ratio, 200M params) ⭐

## CURRENT SCRAPER STATUS

**Active**: Industrial scraper (PID in industrial_scraper.pid)
**Output**: scraped_data_industrial/
**Rate**: Collecting continuously
**Monitor**: tail -f industrial_scraper.log

## CHECKPOINTS

Every 3 days, we'll:
1. Filter collected data
2. Calculate token count
3. Assess model size viability
4. Adjust scraper if needed

## WHEN TO START TRAINING

| Checkpoint | Tokens | Model Size | Quality |
|------------|--------|------------|---------|
| Week 1 | 50-100M | 5-10M params | Small but viable |
| Week 2 | 200-300M | 20-30M params | Medium |
| **Week 3** | **500M-1B** | **100-200M params** | **First 200M viable (risky)** |
| **Week 4** | **1-2B** | **200M params** | **Good (recommended)** ✅ |
| Month 2 | 2-4B | 200M params | Perfect (ideal) ⭐ |

## RECOMMENDATION

**Minimum timeline for 200M with acceptable quality: 3-4 weeks**

We're now running industrial-scale collection. The scraper will run continuously, and I'll check in every few days to filter and report progress.

## INVESTOR PITCH

**What we're building:**
- 200M parameter enterprise pentesting AI
- Trained on 2-4 billion security tokens
- Zero overfitting/underfitting
- Industry-leading quality

**Timeline:**
- Month 1: Data collection (2B tokens)
- Week 5: Tokenizer training
- Weeks 6-7: Model training (200M params)
- Week 8: Fine-tuning and evaluation

**Total**: 2 months to production-ready 200M model
