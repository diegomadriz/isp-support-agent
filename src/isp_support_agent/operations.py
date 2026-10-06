"""Small operational ledger: customer lockouts, staff alerts and webhook receipts."""

import hashlib
import json
import sqlite3
import time
from threading import RLock


class OperationStore:
    def __init__(self, path=None, *, clock=time.time, cooldown=900):
        self.clock, self.cooldown = clock, cooldown
        self.lock = RLock()
        self.connection = sqlite3.connect(
            str(path) if path else ":memory:", check_same_thread=False
        )
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS verification_budget (customer INTEGER PRIMARY KEY, failures INTEGER NOT NULL, until REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS staff_alerts (key TEXT PRIMARY KEY, customer INTEGER, kind TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS inbound_messages (id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL, reply TEXT);
            CREATE TABLE IF NOT EXISTS conversation_replies (sender_hash TEXT PRIMARY KEY);
        """)
        self.connection.commit()

    def budget(self, customer):
        with self.lock:
            row = self.connection.execute(
                "SELECT failures, until FROM verification_budget WHERE customer=?", (customer,)
            ).fetchone()
            if not row or (row[1] and row[1] <= self.clock()):
                return 0, False
            return row[0], row[0] > 0 and row[1] > self.clock()

    def verify(self, customer, matches):
        with self.lock:
            failures, locked = self.budget(customer)
            if locked:
                return False, 0
            failures = 0 if matches else failures + 1
            until = self.clock() + self.cooldown if failures >= 3 else 0
            self.connection.execute(
                "INSERT OR REPLACE INTO verification_budget VALUES (?, ?, ?)",
                (customer, failures, until),
            )
            self.connection.commit()
            return bool(matches), max(0, 3 - failures)

    def alert(self, key, customer=None, kind="critical"):
        with self.lock:
            self.connection.execute(
                "INSERT OR IGNORE INTO staff_alerts VALUES (?, ?, ?)", (key, customer, kind)
            )
            self.connection.commit()

    def critical_alert(self, request_id, sender, customer=None):
        # Anonymous reports collapse per sender per minute; no phone/sender is stored.
        key = (
            request_id
            if customer is not None
            else (
                f"anonymous:{hashlib.sha256(sender.encode()).hexdigest()}:{int(self.clock() // 60)}"
            )
        )
        self.alert(key, customer)

    def alerts(self):
        with self.lock:
            return [
                dict(key=r[0], customer=r[1], kind=r[2])
                for r in self.connection.execute("SELECT * FROM staff_alerts")
            ]

    def unclassified_alert(self, request_id, sender, customer=None):
        key = (
            request_id
            if customer is not None
            else (
                f"anonymous:{hashlib.sha256(sender.encode()).hexdigest()}:{int(self.clock() // 60)}"
            )
        )
        self.alert(key, customer, "unclassified message, model unavailable")

    def has_replied(self, sender):
        with self.lock:
            return (
                self.connection.execute(
                    "SELECT 1 FROM conversation_replies WHERE sender_hash=?",
                    (hashlib.sha256(sender.encode()).hexdigest(),),
                ).fetchone()
                is not None
            )

    def mark_replied(self, sender):
        with self.lock:
            self.connection.execute(
                "INSERT OR IGNORE INTO conversation_replies VALUES (?)",
                (hashlib.sha256(sender.encode()).hexdigest(),),
            )
            self.connection.commit()

    def receive(self, mid, sender, text):
        with self.lock:
            existed = self.connection.execute(
                "SELECT 1 FROM inbound_messages WHERE id=?", (mid,)
            ).fetchone()
            if existed:
                return False
            self.connection.execute(
                "INSERT INTO inbound_messages (id,payload,status) VALUES (?, ?, ?)",
                (mid, json.dumps([sender, text]), "pending"),
            )
            self.connection.commit()
            return True

    def pending(self):
        with self.lock:
            return [
                (mid, *json.loads(payload))
                for mid, payload in self.connection.execute(
                    "SELECT id, payload FROM inbound_messages WHERE status='pending'"
                )
            ]

    def cached_reply(self, mid):
        with self.lock:
            row = self.connection.execute(
                "SELECT reply FROM inbound_messages WHERE id=?", (mid,)
            ).fetchone()
            return row[0] if row else None

    def save_reply(self, mid, text):
        with self.lock:
            self.connection.execute("UPDATE inbound_messages SET reply=? WHERE id=?", (text, mid))
            self.connection.commit()

    def delivered(self, mid):
        with self.lock:
            self.connection.execute(
                "UPDATE inbound_messages SET status='delivered', payload='[]' WHERE id=?", (mid,)
            )
            self.connection.commit()

    def forget(self, mid):
        with self.lock:
            self.connection.execute(
                "DELETE FROM inbound_messages WHERE id=? AND status='pending'", (mid,)
            )
            self.connection.commit()

    def close(self):
        self.connection.close()
