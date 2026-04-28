"""
Tests for the gold price tracker.
"""

import os
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

import tracker


@pytest.fixture(autouse=True)
def use_temp_db(monkeypatch, tmp_path):
    """Redirect the tracker's DB to a temp file for each test."""
    temp_db = tmp_path / "test_tracker.db"
    monkeypatch.setattr(tracker, "DB_PATH", str(temp_db))


def test_init_db_creates_table():
    tracker.init_db()
    conn = sqlite3.connect(tracker.DB_PATH)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='prices'"
    ).fetchone()
    conn.close()
    assert row is not None


def test_record_and_retrieve_price():
    tracker.init_db()
    tracker.record_price(4649.65)
    history = tracker.get_history(limit=1)
    assert len(history) == 1
    assert history[0][0] == 4649.65


def test_get_history_respects_limit():
    tracker.init_db()
    for i in range(20):
        tracker.record_price(1000.0 + i)
    history = tracker.get_history(limit=5)
    assert len(history) == 5


def test_fetch_price_parses_response():
    fake_response = {
        "price": 4649.65,
        "currency": "USD",
        "metal": "XAU",
    }

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return fake_response

    with patch.dict(os.environ, {"GOLDAPI_KEY": "test-key"}):
        with patch("tracker.requests.get", return_value=FakeResp()):
            price = tracker.fetch_price()
    assert price == 4649.65