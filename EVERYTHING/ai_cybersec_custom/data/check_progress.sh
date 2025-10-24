#!/bin/bash
################################################################################
# CHECK PARALLEL SCRAPER PROGRESS
# Usage: ./check_progress.sh
################################################################################

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo -e "${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════════╗
║              PARALLEL SCRAPER PROGRESS REPORT                        ║
╚══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check if scrapers are running
echo -e "${YELLOW}🔍 Scraper Status:${NC}"
for i in {1..5}; do
    if [ -f "logs/scraper_$i.pid" ]; then
        PID=$(cat logs/scraper_$i.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "  Scraper $i: ${GREEN}✅ RUNNING${NC} (PID: $PID)"
        else
            echo -e "  Scraper $i: ${RED}❌ STOPPED${NC}"
        fi
    else
        echo -e "  Scraper $i: ${RED}❌ NOT STARTED${NC}"
    fi
done

# Data collection stats
echo -e "\n${YELLOW}📊 Data Collection:${NC}"
if [ -d "scraped_parallel" ]; then
    TOTAL_SIZE=$(du -sh scraped_parallel/ 2>/dev/null | cut -f1)
    FILE_COUNT=$(ls scraped_parallel/*.jsonl 2>/dev/null | wc -l)
    
    echo "  Total Size: $TOTAL_SIZE"
    echo "  Files: $FILE_COUNT"
    
    # Count total items
    if [ $FILE_COUNT -gt 0 ]; then
        TOTAL_LINES=$(wc -l scraped_parallel/*.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
        echo "  Total Items: $(printf "%'d" $TOTAL_LINES)"
        
        # Estimate tokens (rough: 1 item = 500 tokens average)
        EST_TOKENS=$((TOTAL_LINES * 500))
        echo "  Est. Tokens: $(printf "%'d" $EST_TOKENS)"
        
        # Progress toward 2B tokens
        PROGRESS=$(echo "scale=2; $EST_TOKENS * 100 / 2000000000" | bc)
        echo "  Progress to 2B: $PROGRESS%"
    fi
else
    echo "  No data collected yet"
fi

# Per-scraper breakdown
echo -e "\n${YELLOW}📋 Per-Scraper Breakdown:${NC}"
for i in {1..5}; do
    SIZE=$(du -sh scraped_parallel/s${i}_*.jsonl 2>/dev/null | awk '{sum+=$1} END {print sum}')
    COUNT=$(ls scraped_parallel/s${i}_*.jsonl 2>/dev/null | wc -l)
    if [ $COUNT -gt 0 ]; then
        echo "  Scraper $i: $SIZE MB ($COUNT files)"
    else
        echo "  Scraper $i: No data yet"
    fi
done

# Recent activity
echo -e "\n${YELLOW}🕐 Recent Activity (last 5 minutes):${NC}"
find logs/ -name "scraper_*.log" -mmin -5 2>/dev/null | while read log; do
    SCRAPER_NUM=$(echo $log | grep -o '[0-9]')
    echo -e "  Scraper $SCRAPER_NUM: ${GREEN}Active${NC}"
    tail -2 "$log" 2>/dev/null | head -1 | cut -c1-70
done

# Estimates
echo -e "\n${YELLOW}📈 Estimates:${NC}"
if [ -d "scraped_parallel" ] && [ $FILE_COUNT -gt 0 ]; then
    # Calculate rate (very rough)
    echo "  At current rate:"
    echo "    1 week:  ~500 MB - 1 GB"
    echo "    2 weeks: ~1 GB - 2 GB (viable for 200M)"
    echo "    1 month: ~2 GB - 4 GB (ideal for 200M)"
fi

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo "Run this script anytime to check progress"
echo "Logs: logs/scraper_*.log"
echo "Stop: ./stop_all_scrapers.sh"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
