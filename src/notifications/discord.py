"""Discord notifications — sends trade alerts and daily summaries via webhook.

Messages:
1. Trade opened: instrument, direction, entry price, size, strategy type
2. Trade closed: PnL, exit reason, bars held, capture rate
3. Daily summary: trades, P&L, win rate, drawdown status
4. System alerts: model retrained, errors, combine progress
"""

from __future__ import annotations

import os
from datetime import datetime

import httpx
from loguru import logger


WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


def send_discord(content: str, webhook_url: str = "") -> bool:
    """Send a message to Discord via webhook."""
    url = webhook_url or WEBHOOK_URL
    if not url:
        return False

    try:
        resp = httpx.post(url, json={"content": content}, timeout=10)
        return resp.status_code in (200, 204)
    except Exception as e:
        logger.warning(f"Discord send failed: {e}")
        return False


def send_embed(title: str, description: str, color: int = 0x00FF00, fields: list[dict] | None = None, webhook_url: str = "") -> bool:
    """Send a rich embed message to Discord."""
    url = webhook_url or WEBHOOK_URL
    if not url:
        return False

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "Quant Bot"},
    }
    if fields:
        embed["fields"] = fields

    try:
        resp = httpx.post(url, json={"embeds": [embed]}, timeout=10)
        return resp.status_code in (200, 204)
    except Exception as e:
        logger.warning(f"Discord embed failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# Pre-built notification templates
# ═══════════════════════════════════════════════════════════════════

def notify_trade_opened(
    instrument: str, direction: str, entry_price: float,
    size: int, strategy: str, quality_score: float = 0,
) -> bool:
    """Notify when a trade is opened."""
    emoji = "🟢" if direction == "LONG" else "🔴"
    color = 0x00FF00 if direction == "LONG" else 0xFF0000

    return send_embed(
        title=f"{emoji} {direction} {instrument}",
        description=f"**{strategy}** signal",
        color=color,
        fields=[
            {"name": "Entry", "value": f"${entry_price:,.2f}", "inline": True},
            {"name": "Size", "value": f"{size} contracts", "inline": True},
            {"name": "Quality", "value": f"{quality_score:.0%}", "inline": True},
        ],
    )


def notify_trade_closed(
    instrument: str, direction: str, pnl: float,
    exit_reason: str, bars_held: int, entry_price: float, exit_price: float,
) -> bool:
    """Notify when a trade is closed."""
    emoji = "💰" if pnl > 0 else "💸"
    color = 0x00FF00 if pnl > 0 else 0xFF0000
    pnl_str = f"+${pnl:,.2f}" if pnl > 0 else f"-${abs(pnl):,.2f}"

    return send_embed(
        title=f"{emoji} Closed {direction} {instrument}: {pnl_str}",
        description=f"Exit: **{exit_reason}**",
        color=color,
        fields=[
            {"name": "Entry", "value": f"${entry_price:,.2f}", "inline": True},
            {"name": "Exit", "value": f"${exit_price:,.2f}", "inline": True},
            {"name": "Held", "value": f"{bars_held * 5} min", "inline": True},
        ],
    )


def notify_daily_summary(
    date: str, trades: int, pnl: float, win_rate: float,
    day_drawdown: float, total_pnl: float, combine_progress: float = 0,
) -> bool:
    """Send end-of-day summary."""
    emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➖"
    color = 0x00FF00 if pnl > 0 else 0xFF0000 if pnl < 0 else 0x808080
    pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

    fields = [
        {"name": "Trades", "value": str(trades), "inline": True},
        {"name": "P&L", "value": pnl_str, "inline": True},
        {"name": "Win Rate", "value": f"{win_rate:.0%}", "inline": True},
        {"name": "Day DD", "value": f"${day_drawdown:,.2f}", "inline": True},
        {"name": "Total P&L", "value": f"${total_pnl:,.2f}", "inline": True},
    ]

    if combine_progress > 0:
        fields.append({
            "name": "Combine Progress",
            "value": f"{combine_progress:.0%} of $3,000",
            "inline": True,
        })

    return send_embed(
        title=f"{emoji} Daily Summary — {date}",
        description=f"**{pnl_str}** on {trades} trades",
        color=color,
        fields=fields,
    )


def notify_system_alert(message: str, level: str = "info") -> bool:
    """Send system alert (errors, retraining, etc)."""
    colors = {"info": 0x3498DB, "warning": 0xF39C12, "error": 0xE74C3C, "success": 0x2ECC71}
    emojis = {"info": "ℹ️", "warning": "⚠️", "error": "🚨", "success": "✅"}

    return send_embed(
        title=f"{emojis.get(level, 'ℹ️')} System Alert",
        description=message,
        color=colors.get(level, 0x3498DB),
    )


def notify_walkforward_result(
    instrument: str, window_id: int, test_period: str,
    trades: int, pnl: float, sharpe: float,
) -> bool:
    """Notify walk-forward window result."""
    emoji = "✅" if pnl > 0 else "❌"
    color = 0x00FF00 if pnl > 0 else 0xFF0000

    return send_embed(
        title=f"{emoji} WF #{window_id} — {instrument}",
        description=f"Test: {test_period}",
        color=color,
        fields=[
            {"name": "Trades", "value": str(trades), "inline": True},
            {"name": "P&L", "value": f"${pnl:,.2f}", "inline": True},
            {"name": "Sharpe", "value": f"{sharpe:.2f}", "inline": True},
        ],
    )
