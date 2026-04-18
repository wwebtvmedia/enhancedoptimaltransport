#!/bin/bash
# storage_manager.sh - Resilient storage cleanup for tight disk space environments (Vast.ai)
# This script ensures the training directory never exceeds a safe threshold.

# --- CONFIGURATION ---
LOG_DIR="enhanced_label_sb/logs"
SAMPLE_DIR="enhanced_label_sb/samples"
SNAP_DIR="enhanced_label_sb/snapshots"
CKPT_DIR="enhanced_label_sb/checkpoints"

# Keep counts (Adjust based on disk space)
KEEP_LOGS=5
KEEP_SAMPLES=10
KEEP_SNAPS=3
KEEP_CKPTS=2

INTERVAL=300 # Run every 5 minutes

echo "🧹 Storage Manager started..."
echo "   Keeping: $KEEP_LOGS logs, $KEEP_SAMPLES samples, $KEEP_SNAPS snapshots"

# Function to safely delete oldest files in a directory
cleanup_dir() {
    local target_dir=$1
    local keep_count=$2
    local pattern=$3

    if [ -d "$target_dir" ]; then
        # Count matching files
        local file_count=$(ls -1 $target_dir/$pattern 2>/dev/null | wc -l)
        
        if [ "$file_count" -gt "$keep_count" ]; then
            local to_remove=$((file_count - keep_count))
            echo "[$(date)] 🗑️ Removing $to_remove oldest files from $target_dir..."
            
            # Sort by time (oldest first) and delete
            ls -tr $target_dir/$pattern | head -n "$to_remove" | xargs -I {} rm -f "{}"
        fi
    fi
}

# Trap signals to ensure clean exit if needed, but the loop is designed to be infinite
trap "echo 'Stopping storage manager...'; exit" SIGINT SIGTERM

while true; do
    # 1. Cleanup old training logs (the .pt files in logs/ can be huge)
    cleanup_dir "$LOG_DIR" "$KEEP_LOGS" "*.pt"
    cleanup_dir "$LOG_DIR" "$KEEP_LOGS" "*.log"

    # 2. Cleanup old visual samples (prevent 1000s of PNGs)
    cleanup_dir "$SAMPLE_DIR" "$KEEP_SAMPLES" "*.png"

    # 3. Cleanup old snapshots
    cleanup_dir "$SNAP_DIR" "$KEEP_SNAPS" "*.pt"

    # 4. Emergency Disk Check (if disk > 90% full, aggressive cleanup)
    DISK_USAGE=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 90 ]; then
        echo "⚠️ CRITICAL: Disk usage at ${DISK_USAGE}%. Running aggressive cleanup..."
        # Delete ALL samples except last 2
        cleanup_dir "$SAMPLE_DIR" 2 "*.png"
        # Delete ALL logs except last 2
        cleanup_dir "$LOG_DIR" 2 "*.pt"
        # Sync to free up blocks
        sync
    fi

    sleep "$INTERVAL"
done
