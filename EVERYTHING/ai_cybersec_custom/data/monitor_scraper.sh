#!/bin/bash
# Monitor the comprehensive scraper progress

echo "=========================================="
echo "📊 SCRAPER MONITORING"
echo "=========================================="
echo ""

# Check if scraper is running
if ps aux | grep -v grep | grep "mega_scraper.py" > /dev/null; then
    echo "✅ Scraper Status: RUNNING"
    echo "   PID: $(ps aux | grep -v grep | grep "mega_scraper.py" | awk '{print $2}')"
else
    echo "❌ Scraper Status: NOT RUNNING"
fi

echo ""
echo "📂 Data Collected:"
FILE_COUNT=$(ls scraped_data/*.jsonl 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh scraped_data 2>/dev/null | cut -f1)
echo "   Files: $FILE_COUNT"
echo "   Total size: $TOTAL_SIZE"

echo ""
echo "📝 Recent Log (last 15 lines):"
tail -15 comprehensive_scraper_output.log 2>/dev/null || tail -15 mega_scraper.log 2>/dev/null || echo "No log file found"

echo ""
echo "=========================================="
