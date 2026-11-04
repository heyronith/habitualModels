from __future__ import annotations

import secrets
from datetime import datetime, timezone


def make_run_id(name: str) -> str:
    """YYYYMMDD-HHMMSS-<name>-<6hex> in UTC."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    hex6 = secrets.token_hex(3)
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)[:48]
    return f"{stamp}-{safe}-{hex6}"
