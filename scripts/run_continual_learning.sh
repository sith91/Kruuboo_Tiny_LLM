#!/usr/bin/env bash
# run_continual_learning.sh - Launches the Kruboo continual learning daemon in the background.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
PID_FILE="$PROJECT_DIR/continual_learning.pid"
LOG_FILE="$PROJECT_DIR/continual_learning.log"

start_daemon() {
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null; then
      echo "⚠️  Continual learning daemon is already running (PID: $PID)."
      exit 1
    else
      rm -f "$PID_FILE"
    fi
  fi

  echo "🚀 Starting Kruboo Continual Learning Daemon..."
  nohup python3 "$SCRIPT_DIR/continual_daemon.py" > "$LOG_FILE" 2>&1 &
  DAEMON_PID=$!
  
  echo "$DAEMON_PID" > "$PID_FILE"
  echo "✅ Daemon started in background with PID: $DAEMON_PID"
  echo "📝 Logs are being written to: $LOG_FILE"
}

stop_daemon() {
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null; then
      echo "🛑 Stopping Kruboo Continual Learning Daemon (PID: $PID)..."
      kill "$PID"
      rm -f "$PID_FILE"
      echo "✅ Daemon stopped."
    else
      echo "⚠️  Daemon process not running. Cleaning up stale PID file."
      rm -f "$PID_FILE"
    fi
  else
    echo "⚠️  No PID file found. Is the daemon running?"
  fi
}

status_daemon() {
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null; then
      echo "🟢 Daemon is running (PID: $PID)."
      echo "📋 Last 10 lines of logs:"
      tail -n 10 "$LOG_FILE"
    else
      echo "🔴 Daemon is not running (stale PID file)."
    fi
  else
    echo "🔴 Daemon is not running."
  fi
}

case "$1" in
  start)
    start_daemon
    ;;
  stop)
    stop_daemon
    ;;
  status)
    status_daemon
    ;;
  *)
    echo "Usage: $0 {start|stop|status}"
    exit 1
    ;;
esac
