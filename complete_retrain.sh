#!/bin/bash
set -e  # Exit on error

echo "========================================"
echo "  COMPLETE MODEL RETRAIN - OPTIMIZED"
echo "========================================"
echo ""

# 1. Update config
echo "✅ Step 1: Updating config to ULTRA_TINY model..."
echo "   Manual step: Replace ai_cybersec_custom/utils/config.py"
echo "   with 'Optimized Config for Small Data' artifact"
echo ""
read -p "Press Enter when config.py is updated..."

# 2. Update transformer
echo ""
echo "✅ Step 2: Updating to simplified transformer..."
echo "   Manual step: Replace ai_cybersec_custom/model/custom_transformer.py"
echo "   with 'Simplified Transformer for Small Data' artifact"
echo ""
read -p "Press Enter when custom_transformer.py is updated..."

# 3. Update corpus
echo ""
echo "✅ Step 3: Updating corpus to expanded version..."
echo "   Manual step: Replace ai_cybersec_custom/data/corpus.txt"
echo "   with 'Expanded Training Corpus' artifact"
echo ""
read -p "Press Enter when corpus.txt is updated..."

# 4. Clean old files
echo ""
echo "🗑️  Step 4: Cleaning old tokenizer and checkpoint..."
rm -f ai_cybersec_custom/tokenizer/bpe.model
rm -f ai_cybersec_custom/tokenizer/bpe.vocab
rm -f ai_cybersec_custom/train/utils/checkpoint.pt
echo "   ✅ Old files deleted"

# 5. Retrain tokenizer
echo ""
echo "📝 Step 5: Training new tokenizer (vocab_size=2000)..."
cd ai_cybersec_custom/tokenizer
python3 train_tokenizer.py
if [ $? -ne 0 ]; then
    echo "❌ Tokenizer training failed!"
    exit 1
fi
echo "   ✅ Tokenizer trained"

# 6. Verify tokenizer
echo ""
echo "🔍 Step 6: Verifying tokenizer..."
cd ../..
python3 verify_tokenizer.py
if [ $? -ne 0 ]; then
    echo "❌ Tokenizer verification failed!"
    exit 1
fi
echo "   ✅ Tokenizer verified"

# 7. Train model
echo ""
echo "🚀 Step 7: Training model (100 epochs)..."
echo "   This will take 10-30 minutes depending on your hardware..."
echo "   Watch for loss dropping below 2.0"
echo ""
cd ai_cybersec_custom/train
python3 train.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi
echo "   ✅ Training complete"

# 8. Success
echo ""
echo "========================================"
echo "  ✅ ALL DONE!"
echo "========================================"
echo ""
echo "Your model is ready to test!"
echo ""
echo "To chat with it:"
echo "  cd ../.."
echo "  python3 ai_cybersec_custom/chat.py"
echo ""
echo "Expected results:"
echo "  ✅ Basic sentence structure"
echo "  ✅ Can answer questions from training"
echo "  ✅ Some generalization to similar questions"
echo "  ⚠️  May still repeat phrases for novel questions"
echo ""
echo "Model info:"
echo "  - Parameters: ~100K (vs 8M before)"
echo "  - Training samples: 800+ (vs 150 before)"
echo "  - Epochs: 100"
echo "  - Architecture: Simplified transformer"
echo ""