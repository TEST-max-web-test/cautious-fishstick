#!/bin/bash
################################################################################
# STOP ALL PARALLEL SCRAPERS
# Usage: ./stop_all_scrapers.sh
################################################################################

echo "🛑 Stopping all parallel scrapers..."

# Kill by pattern
pkill -f "parallel_scraper_" 2>/dev/null

# Kill by PID files
for i in {1..5}; do
    if [ -f "logs/scraper_$i.pid" ]; then
        PID=$(cat logs/scraper_$i.pid)
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null
            echo "  ✅ Stopped scraper $i (PID: $PID)"
        fi
    fi
done

sleep 2

# Verify
RUNNING=$(ps aux | grep parallel_scraper | grep -v grep | wc -l)
if [ $RUNNING -eq 0 ]; then
    echo "✅ All scrapers stopped successfully"
else
    echo "⚠️  $RUNNING scrapers still running. Force killing..."
    pkill -9 -f "parallel_scraper_"
    echo "✅ Force stopped"
fi

# Show final stats
echo ""
echo "📊 Final Collection Stats:"
du -sh scraped_parallel/ 2>/dev/null || echo "No data collected yet"
echo ""
echo "Files: $(ls scraped_parallel/*.jsonl 2>/dev/null | wc -l)"
echo ""
echo "To restart: ./start_all_scrapers.sh"
