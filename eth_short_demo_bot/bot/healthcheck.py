from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--max-heartbeat-age-seconds", type=int, default=90)
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(1)

    conn = sqlite3.connect(db)
    row = conn.execute("SELECT value FROM kv_state WHERE key = 'heartbeat'").fetchone()
    conn.close()
    if row is None:
        raise SystemExit(1)
    import json

    payload = json.loads(row[0])
    ts = datetime.fromisoformat(payload["ts"])
    age = datetime.now(timezone.utc) - ts
    if age > timedelta(seconds=args.max_heartbeat_age_seconds):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
