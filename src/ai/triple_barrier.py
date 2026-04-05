"""Triple-barrier labeling — labels trades by actual outcome quality, not just win/loss.

Instead of binary (profitable = 1, losing = 0), compute:
- R-multiple: how many R of risk the trade captured
- Barrier hit: TP / SL / timeout / trailing / session_flatten
- Continuous label that the EV regressor can learn from

A trade that hits 2R TP gets label 2.0
A trade that trails out at 1.5R gets label 1.5
A trade that times out at 0.3R gets label 0.3
A trade that hits SL gets label -1.0

This gives ML models much richer training signal than binary classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_r_multiple(trade) -> float:
    """Compute R-multiple for a completed trade.

    R = net_pnl / risk_per_trade
    Where risk = |entry - stop| × size × tick_value / tick_size

    Simplified: we use the raw PnL ratio since stop distance
    is already factored into position sizing.
    """
    net = trade.pnl - trade.fees
    # Risk = distance to stop × size × tick_value / tick_size
    stop_dist = abs(trade.entry_price - trade.sl_price)
    if stop_dist == 0:
        return 0.0

    # R = actual move / stop distance (in price terms)
    actual_move = (trade.exit_price - trade.entry_price) * trade.direction
    r_multiple = actual_move / stop_dist

    return float(r_multiple)


def label_trades_triple_barrier(trades: list, tick_value: float = 0.50, tick_size: float = 0.25) -> pd.DataFrame:
    """Label a list of Trade objects with triple-barrier outcomes.

    Returns DataFrame with:
    - entry_bar, net_pnl (standard)
    - r_multiple: continuous R label
    - barrier_type: which exit barrier was hit
    - label_class: categorical (strong_win / weak_win / scratch / weak_loss / strong_loss)
    """
    records = []
    for t in trades:
        net = t.pnl - t.fees
        r_mult = compute_r_multiple(t)
        bars_held = t.exit_bar - t.entry_bar if t.exit_bar else 0

        # Classify barrier type
        barrier = t.exit_reason

        # Classify quality
        if r_mult >= 1.5:
            label_class = "strong_win"
        elif r_mult >= 0.5:
            label_class = "weak_win"
        elif r_mult >= -0.3:
            label_class = "scratch"
        elif r_mult >= -0.8:
            label_class = "weak_loss"
        else:
            label_class = "strong_loss"

        records.append({
            "entry_bar": t.entry_bar,
            "net_pnl": net,
            "pnl": t.pnl,
            "fees": t.fees,
            "r_multiple": r_mult,
            "barrier_type": barrier,
            "label_class": label_class,
            "direction": t.direction,
            "size": t.size,
            "bars_held": bars_held,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "sl_price": t.sl_price,
            "tp_price": t.tp_price,
            "peak_profit": t.peak_profit,
        })

    return pd.DataFrame(records)
