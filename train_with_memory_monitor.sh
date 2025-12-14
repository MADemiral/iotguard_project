#!/bin/bash

# Memory-efficient training script for IoTGuard
# Monitors memory usage and prevents OOM kills

echo "=== IoTGuard Training with Memory Monitoring ==="
echo "Starting at: $(date)"
echo ""

# Clear system cache
echo "Clearing system cache..."
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Show initial memory
echo ""
echo "Initial memory status:"
free -h
echo ""

# Set memory limits for Python
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Limit GPU memory growth for TensorFlow (even though we use CPU)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Run training with memory monitoring in background
python scripts/training/train_ultra_advanced.py &
TRAIN_PID=$!

echo "Training started with PID: $TRAIN_PID"
echo "Monitoring memory usage..."
echo ""

# Monitor memory every 30 seconds
while kill -0 $TRAIN_PID 2>/dev/null; do
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", ($3/$2) * 100}')
    SWAP_USAGE=$(free | grep Swap | awk '{printf "%.1f", ($3/$2) * 100}')
    
    echo "[$(date +%H:%M:%S)] Memory: ${MEMORY_USAGE}% | Swap: ${SWAP_USAGE}%"
    
    # Warning if memory usage is high
    if (( $(echo "$MEMORY_USAGE > 85.0" | bc -l) )); then
        echo "WARNING: High memory usage detected!"
    fi
    
    sleep 30
done

# Wait for training to complete
wait $TRAIN_PID
EXIT_CODE=$?

echo ""
echo "Training completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""
echo "Final memory status:"
free -h

exit $EXIT_CODE
