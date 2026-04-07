#!/bin/bash
# Launch live trading: MNQ (stats-based) + MGC (AI-blocked)
# Run from /root/trader on the droplet

cd /root/trader
source .venv/bin/activate
export PYTHONPATH=/root/trader

# Find latest models
MNQ_STATS=$(ls -t data/models/strategy_stats*.pkl 2>/dev/null | head -1)
MGC_AI=$(ls -t data/models/mgc_hyb_ai_*.pkl 2>/dev/null | head -1)
MGC_STATS=$(ls -t data/models/mgc_hyb_stats_*.pkl 2>/dev/null | head -1)

echo "MNQ stats bank: $MNQ_STATS"
echo "MGC AI model:   $MGC_AI"
echo "MGC stats bank: $MGC_STATS"

# Kill existing bots
pkill -f "live_bot.py" 2>/dev/null
sleep 2

# MNQ: stats-based (no AI blocking)
echo "Starting MNQ bot..."
screen -dmS mnq bash -c "cd /root/trader && source .venv/bin/activate && PYTHONPATH=. python3 -m src.execution.live_bot \
    --instrument MNQ \
    --contract CON.F.US.MNQ.M26 \
    --stats-bank '$MNQ_STATS' \
    > /root/mnq_live.log 2>&1"

# MGC: AI-blocked (use_ai + stats bank for exits)
echo "Starting MGC bot..."
screen -dmS mgc bash -c "cd /root/trader && source .venv/bin/activate && PYTHONPATH=. python3 -m src.execution.live_bot \
    --instrument MGC \
    --contract CON.F.US.MGC.M26 \
    --use-ai \
    --ai-model '$MGC_AI' \
    --stats-bank '$MGC_STATS' \
    > /root/mgc_live.log 2>&1"

sleep 2
echo ""
echo "Running bots:"
screen -ls | grep -E "mnq|mgc"
echo ""
echo "Logs: tail -f /root/mnq_live.log /root/mgc_live.log"
