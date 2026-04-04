#!/bin/bash
# vast_start.sh - Master entrypoint for training and checkpoint persistence

# 1. Kill any existing tmux sessions with the same name
tmux kill-session -t sb_train 2>/dev/null

# 2. Start the checkpoint backup watcher in the background
# It will run as a detached background process
nohup ./checkpoint_backup.sh > backup.log 2>&1 &
BACKUP_PID=$!
echo "🛡️ Checkpoint backup watcher started (PID: $BACKUP_PID)"

# 3. Create a new tmux session and run the training
echo "🚀 Launching training in tmux session 'sb_train'..."
tmux new-session -d -s sb_train "python main.py"

echo "✅ All processes started!"
echo "----------------------------------------------------"
echo "  - To see training logs: tmux attach -t sb_train"
echo "  - To see backup logs:   tail -f backup.log"
echo "  - Backups saved in:     /workspace/backup_checkpoints/"
echo "----------------------------------------------------"
