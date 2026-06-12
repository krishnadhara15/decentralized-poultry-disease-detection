"""Append-only blockchain ledger for tamper-proof flock health records."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config import LEDGER_DIR


@dataclass
class HealthRecord:
    farm_id: str
    image_path: str
    label: str
    confidence: float
    alert_triggered: bool
    timestamp: float
    metadata: dict[str, Any]


@dataclass
class Block:
    index: int
    timestamp: float
    record: HealthRecord
    previous_hash: str
    hash: str


class PoultryLedger:
    """
    Scoped blockchain: multiple independent farms share one append-only ledger.
    Each block = one diagnosis; hash chain detects tampering.
    Single-owner deployment → use a normal DB instead.
    """

    def __init__(self, ledger_path: Path | None = None):
        self.path = ledger_path or LEDGER_DIR / "chain.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.chain: list[Block] = []
        self._load_or_genesis()

    @staticmethod
    def _hash_block(index: int, timestamp: float, record: dict, previous_hash: str) -> str:
        payload = json.dumps(
            {"index": index, "timestamp": timestamp, "record": record, "previous_hash": previous_hash},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _genesis(self) -> Block:
        record = HealthRecord(
            farm_id="system",
            image_path="",
            label="genesis",
            confidence=1.0,
            alert_triggered=False,
            timestamp=time.time(),
            metadata={"note": "Poultry health ledger initialized"},
        )
        block = Block(
            index=0,
            timestamp=record.timestamp,
            record=record,
            previous_hash="0" * 64,
            hash="",
        )
        block.hash = self._hash_block(0, block.timestamp, asdict(record), block.previous_hash)
        return block

    def _load_or_genesis(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.chain = [
                Block(
                    index=b["index"],
                    timestamp=b["timestamp"],
                    record=HealthRecord(**b["record"]),
                    previous_hash=b["previous_hash"],
                    hash=b["hash"],
                )
                for b in raw
            ]
        else:
            self.chain = [self._genesis()]
            self._persist()

    def _persist(self) -> None:
        serializable = [
            {
                "index": b.index,
                "timestamp": b.timestamp,
                "record": asdict(b.record),
                "previous_hash": b.previous_hash,
                "hash": b.hash,
            }
            for b in self.chain
        ]
        self.path.write_text(json.dumps(serializable, indent=2))

    def append(self, record: HealthRecord) -> Block:
        prev = self.chain[-1]
        index = prev.index + 1
        ts = record.timestamp or time.time()
        record.timestamp = ts
        block_hash = self._hash_block(index, ts, asdict(record), prev.hash)
        block = Block(index=index, timestamp=ts, record=record, previous_hash=prev.hash, hash=block_hash)
        self.chain.append(block)
        self._persist()
        return block

    def verify(self) -> bool:
        """Walk the chain; return False if any link is broken."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            expected = self._hash_block(
                current.index,
                current.timestamp,
                asdict(current.record),
                current.previous_hash,
            )
            if current.hash != expected:
                return False
            if current.previous_hash != self.chain[i - 1].hash:
                return False
        return True

    def records_for_farm(self, farm_id: str) -> list[HealthRecord]:
        return [b.record for b in self.chain if b.record.farm_id == farm_id and b.index > 0]

    @property
    def length(self) -> int:
        return len(self.chain)
