#!/bin/bash
# Start the Prompt Forest UI (frontend + backend)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "  ╔═══════════════════════════════════════╗"
echo "  ║       Prompt Forest UI Launcher       ║"
echo "  ╚═══════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "  ✗ python3 not found"
  exit 1
fi

# Check Node
if ! command -v node &>/dev/null; then
  echo "  ✗ node not found — install Node.js 18+"
  exit 1
fi

# Install Python deps if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "  Installing fastapi + uvicorn…"
  pip install fastapi uvicorn 2>&1 | tail -3
fi

# Install frontend deps if needed
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "  Installing frontend dependencies…"
  cd "$ROOT/frontend" && npm install
  cd "$ROOT"
fi

echo "  Starting backend  → http://localhost:8000"
echo "  Starting frontend → http://localhost:3000"
echo ""
echo "  Press Ctrl+C to stop both servers."
echo ""

# Start backend in background
python3 "$ROOT/api_server.py" &
BACKEND_PID=$!

# Start frontend
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

# Trap Ctrl+C
cleanup() {
  echo ""
  echo "  Shutting down…"
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
  echo "  Done."
}
trap cleanup INT TERM

wait $FRONTEND_PID
