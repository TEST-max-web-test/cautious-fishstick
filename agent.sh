#!/bin/bash

################################################################################
# AUTOMATED SETUP SCRIPT
# Copy this entire file to setup.sh and run: bash setup.sh
# This will do everything automatically
################################################################################

set -e  # Exit on error

echo ""
echo "================================================================================"
echo "🚀 ENTERPRISE PENTESTING AI - AUTOMATED SETUP"
echo "================================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ $MAJOR -lt 3 ] || [ $MAJOR -eq 3 -a $MINOR -lt 10 ]; then
    print_error "Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi
print_status "Python $PYTHON_VERSION"

# Step 1: Upgrade pip
echo ""
echo "Step 1: Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_status "pip upgraded"

# Step 2: Create requirements.txt
echo ""
echo "Step 2: Creating requirements.txt..."
cat > requirements.txt << 'EOF'
torch==2.1.2
torchvision==0.16.2
torchaudio==0.16.2
sentencepiece==0.2.0
numpy==1.26.4
scikit-learn==1.4.1
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3
psutil==5.9.8
EOF
print_status "requirements.txt created"

# Step 3: Install packages
echo ""
echo "Step 3: Installing packages (this may take a few minutes)..."
pip install -r requirements.txt
print_status "All packages installed"

# Step 4: Verify installations
echo ""
echo "Step 4: Verifying installations..."
python3 -c "import torch; print('   PyTorch:', torch.__version__)" > /dev/null
print_status "PyTorch verified"

python3 -c "import sentencepiece" > /dev/null
print_status "sentencepiece verified"

python3 -c "import numpy; print('   NumPy:', numpy.__version__)" > /dev/null
print_status "NumPy verified"

python3 -c "import fastapi" > /dev/null
print_status "FastAPI verified"

# Step 5: Check corpus
echo ""
echo "Step 5: Checking corpus..."
if [ -f "ai_cybersec_custom/data/corpus.txt" ]; then
    LINES=$(wc -l < ai_cybersec_custom/data/corpus.txt)
    SIZE=$(du -h ai_cybersec_custom/data/corpus.txt | awk '{print $1}')
    print_status "corpus.txt found: $LINES lines, $SIZE"
    if [ $LINES -lt 500 ]; then
        print_warning "Corpus is small (only $LINES lines). Consider replacing with complete_corpus_full from artifacts."
    fi
else
    print_warning "corpus.txt not found. Please copy from complete_corpus_full artifact."
fi

# Step 6: Train tokenizer
echo ""
echo "Step 6: Training tokenizer..."
cd ai_cybersec_custom/tokenizer
python3 train_tokenizer.py
cd ../..
print_status "Tokenizer trained"

# Step 7: Create checkpoint directory
echo ""
echo "Step 7: Creating checkpoint directory..."
mkdir -p ai_cybersec_custom/train/HERE
print_status "Checkpoint directory created"

# Step 8: Test imports
echo ""
echo "Step 8: Testing imports..."
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
from ai_cybersec_custom.model.custom_transformer import ModernTransformer
from ai_cybersec_custom.tokenizer.custom_tokenizer import CustomTokenizer
from ai_cybersec_custom.utils.config import MODEL_CONFIG, TRAIN_CONFIG
import torch

print("   ✅ Model imports")
print("   ✅ Tokenizer imports")
print("   ✅ Config imports")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    prop = torch.cuda.get_device_properties(0)
    print(f"   ✅ GPU: {prop.name} ({prop.total_memory / 1e9:.2f} GB VRAM)")
else:
    print(f"   ✅ CPU (GPU not detected, training will be slow)")
PYEOF

print_status "All imports OK"

# Step 9: Show next steps
echo ""
echo "================================================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Verify corpus size (should be 500KB+):"
echo "   du -h ai_cybersec_custom/data/corpus.txt"
echo ""
echo "2. Start training:"
echo "   python3 ai_cybersec_custom/train/train.py"
echo ""
echo "   This will run for 6-24 hours depending on your GPU."
echo "   You can monitor progress and it will save checkpoints."
echo ""
echo "3. (Optional) Deploy API after training:"
echo "   python3 api.py"
echo ""
echo "   Then access at: http://localhost:8000"
echo "   Docs at: http://localhost:8000/docs"
echo ""
echo "================================================================================"
echo ""