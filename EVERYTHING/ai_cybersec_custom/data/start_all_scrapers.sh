#!/bin/bash
################################################################################
# PARALLEL SCRAPER LAUNCHER FOR 200M PARAMETER MODEL
# Runs 5 scrapers simultaneously to collect 2-4 billion tokens
# 
# Usage: ./start_all_scrapers.sh
# Stop:  ./stop_all_scrapers.sh (or Ctrl+C and run stop script)
################################################################################

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════════╗
║           PARALLEL SCRAPER SYSTEM FOR 200M MODEL                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  5 Scrapers Running Simultaneously                                   ║
║  Target: 2-4 billion tokens (2-4 GB filtered)                        ║
║  Timeline: 1-2 weeks with parallel collection                        ║
║  Output: scraped_parallel/                                           ║
╚══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Navigate to data directory
cd "$(dirname "$0")"

# Kill any existing scrapers
echo -e "${YELLOW}🛑 Stopping any existing scrapers...${NC}"
pkill -f "parallel_scraper_" 2>/dev/null
sleep 2

# Create output directory
mkdir -p scraped_parallel
mkdir -p logs

# Make scrapers executable
chmod +x parallel_scraper_*.py

echo -e "\n${GREEN}🚀 LAUNCHING 5 PARALLEL SCRAPERS...${NC}\n"

# Start each scraper in background
echo -e "${BLUE}[1/5]${NC} Starting OWASP & Web Security scraper..."
nohup python3 -u parallel_scraper_1.py > logs/scraper_1.log 2>&1 &
PID1=$!
echo $PID1 > logs/scraper_1.pid
echo -e "  ✅ PID: $PID1"
sleep 2

echo -e "${BLUE}[2/5]${NC} Starting Pentesting & Red Team scraper..."
nohup python3 -u parallel_scraper_2.py > logs/scraper_2.log 2>&1 &
PID2=$!
echo $PID2 > logs/scraper_2.pid
echo -e "  ✅ PID: $PID2"
sleep 2

echo -e "${BLUE}[3/5]${NC} Starting CTF & Reverse Engineering scraper..."
nohup python3 -u parallel_scraper_3.py > logs/scraper_3.log 2>&1 &
PID3=$!
echo $PID3 > logs/scraper_3.pid
echo -e "  ✅ PID: $PID3"
sleep 2

echo -e "${BLUE}[4/5]${NC} Starting Cloud Security & Vuln Tools scraper..."
nohup python3 -u parallel_scraper_4.py > logs/scraper_4.log 2>&1 &
PID4=$!
echo $PID4 > logs/scraper_4.pid
echo -e "  ✅ PID: $PID4"
sleep 2

echo -e "${BLUE}[5/5]${NC} Starting CVE Database scraper..."
nohup python3 -u parallel_scraper_5.py > logs/scraper_5.log 2>&1 &
PID5=$!
echo $PID5 > logs/scraper_5.pid
echo -e "  ✅ PID: $PID5"

echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ ALL 5 SCRAPERS STARTED SUCCESSFULLY${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}\n"

echo "Process IDs (PIDs):"
echo "  Scraper 1 (OWASP):     $PID1"
echo "  Scraper 2 (Pentest):   $PID2"
echo "  Scraper 3 (CTF):       $PID3"
echo "  Scraper 4 (Cloud):     $PID4"
echo "  Scraper 5 (CVE):       $PID5"

echo -e "\n${YELLOW}📁 Output Directory:${NC} scraped_parallel/"
echo -e "${YELLOW}📋 Log Directory:${NC} logs/"

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}MONITORING COMMANDS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

echo "✅ Check if scrapers are running:"
echo "   ps aux | grep parallel_scraper | grep -v grep"
echo ""
echo "📊 Check progress:"
echo "   ./check_progress.sh"
echo ""
echo "📋 Watch individual scraper logs:"
echo "   tail -f logs/scraper_1.log   # OWASP & Web"
echo "   tail -f logs/scraper_2.log   # Pentesting"
echo "   tail -f logs/scraper_3.log   # CTF & RE"
echo "   tail -f logs/scraper_4.log   # Cloud"
echo "   tail -f logs/scraper_5.log   # CVE"
echo ""
echo "📈 Check total data collected:"
echo "   du -sh scraped_parallel/"
echo ""
echo "🛑 Stop all scrapers:"
echo "   ./stop_all_scrapers.sh"

echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Scrapers will run continuously until stopped manually${NC}"
echo -e "${GREEN}Estimated: 1-2 GB in first week, 2-4 GB in 2 weeks${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}\n"

# Wait a bit and verify
sleep 10
echo -e "${YELLOW}🔍 Verification Check (10 seconds after start):${NC}"
RUNNING=$(ps aux | grep parallel_scraper | grep -v grep | wc -l)
echo "  Scrapers running: $RUNNING / 5"

if [ $RUNNING -eq 5 ]; then
    echo -e "${GREEN}  ✅ All scrapers confirmed running!${NC}"
else
    echo -e "${RED}  ⚠️  Warning: Only $RUNNING scrapers running${NC}"
    echo "  Check logs for errors:"
    echo "    tail logs/scraper_*.log"
fi

echo ""
echo "Happy scraping! 🚀"
