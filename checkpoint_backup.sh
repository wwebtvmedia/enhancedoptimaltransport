#!/bin/bash
# checkpoint_backup.sh - Periodic background backup of latest checkpoint

SOURCE_CKPT="enhanced_label_sb/checkpoints/latest.pt"
BACKUP_DIR="/workspace/backup_checkpoints"
INTERVAL=300 # 5 minutes

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "🛡️ Starting Checkpoint Backup Watcher (Every ${INTERVAL}s)..."
echo "   Source: $SOURCE_CKPT"
echo "   Target: $BACKUP_DIR/"

while true; do
    if [ -f "$SOURCE_CKPT" ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        # Keep only two files: the absolute latest and a rotating backup
        cp "$SOURCE_CKPT" "$BACKUP_DIR/latest_backup.pt"
        # Optional: Keep a timestamped version (be careful with disk space!)
        # cp "$SOURCE_CKPT" "$BACKUP_DIR/ckpt_${TIMESTAMP}.pt"
        echo "[$(date)] ✅ Checkpoint backed up to $BACKUP_DIR/latest_backup.pt"
    else
        echo "[$(date)] ⏳ Waiting for first checkpoint ($SOURCE_CKPT)..."
    fi
    sleep "$INTERVAL"
done
