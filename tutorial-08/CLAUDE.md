# Gold Price Tracker

A small CLI tool that fetches the current gold price from GoldAPI
and logs each check to a local SQLite database. Supports viewing history.

## Stack
- Python 3.10+
- requests (HTTP client)
- sqlite3 (stdlib)
- pytest (test runner)

## Files
- `tracker.py` — the CLI, contains all logic
- `test_tracker.py` — pytest suite
- `requirements.txt` — dependencies

## Commands
- Install: `pip install -r requirements.txt`
- Run tests: `pytest -v`
- Check current price: `python tracker.py check`
- View history: `python tracker.py history --last 10`

## Environment
- GOLDAPI_KEY must be set. Copy from ~/ai-orchestration-lab/.env if needed.
- Tests must not hit the real GoldAPI — mock the requests call.

## Conventions
- Use f-strings, not .format().
- Top-level CLI parsing with argparse.
- Database path should be a local file, not a system path.