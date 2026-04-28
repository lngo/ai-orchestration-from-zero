"""
Gold Price Tracker — CLI for checking and logging gold prices.
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime

import requests

# BUG #2: this path does not exist on a standard Ubuntu user account
# and is not writable without sudo. Claude Code must choose a sensible
# alternative.
DB_PATH = "/var/data/gold_tracker.db"


def fetch_price():
    """Fetch the current gold price in USD per troy ounce from GoldAPI."""
    key = os.environ.get("GOLDAPI_KEY")
    if not key:
        raise RuntimeError("GOLDAPI_KEY not set in environment")
    headers = {"x-access-token": key, "Content-Type": "application/json"}
    r = requests.get("https://www.goldapi.io/api/XAU/USD", headers=headers)
    r.raise_for_status()
    data = r.json()
    # BUG #1: the GoldAPI response uses the key "price", not "gold_price".
    return data["gold_price"]


def init_db():
    """Create the prices table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            price REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def record_price(price):
    """Record a single price observation with a UTC timestamp."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO prices (price, timestamp) VALUES (?, ?)",
        (price, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(limit=10):
    """Return the most recent `limit` observations, newest first."""
    conn = sqlite3.connect(DB_PATH)
    # BUG #3: the SQL query has no LIMIT clause and the `limit` parameter
    # is never used. Returns all rows regardless of what the caller asks for.
    rows = conn.execute(
        "SELECT price, timestamp FROM prices ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


def cmd_check(args):
    init_db()
    price = fetch_price()
    record_price(price)
    print(f"Current gold price: USD {price:.2f} per oz")


def cmd_history(args):
    init_db()
    rows = get_history(limit=args.last)
    if not rows:
        print("No history yet. Run 'check' first.")
        return
    for price, ts in rows:
        print(f"{ts}  USD {price:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Gold Price Tracker")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="Fetch and log current gold price")

    hist = sub.add_parser("history", help="Show price history")
    hist.add_argument("--last", type=int, default=10,
                      help="How many recent entries to show")

    args = parser.parse_args()
    if args.command == "check":
        cmd_check(args)
    elif args.command == "history":
        cmd_history(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()