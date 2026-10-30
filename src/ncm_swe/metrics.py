"""Append-only JSONL metric logger."""

import json
import time
from pathlib import Path


class MetricLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **kv) -> None:
        kv["t"] = time.strftime("%H:%M:%S")
        with open(self.path, "a") as f:
            f.write(json.dumps(kv) + "\n")

    def last(self) -> dict | None:
        if not self.path.exists():
            return None
        lines = self.path.read_text().strip().splitlines()
        return json.loads(lines[-1]) if lines else None
