"""One-off helper used to validate and normalise finetune/data/seed_qa.jsonl.

Not part of the pipeline; kept because it documents the schema and is handy
when the student edits the seed file. Run:
    python finetune/data/_check_seed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent / "seed_qa.jsonl"
ROLES = {"system", "user", "assistant"}


def main() -> int:
    if not PATH.exists():
        print(f"missing {PATH}")
        return 1
    lines = PATH.read_text(encoding="utf-8").splitlines()
    ids, n_msgs = [], 0
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        assert set(obj) == {"messages", "source", "id"}, f"line {i}: keys {set(obj)}"
        assert obj["id"] not in ids, f"line {i}: duplicate id {obj['id']}"
        ids.append(obj["id"])
        assert isinstance(obj["source"], str) and obj["source"], f"line {i}: source"
        assert isinstance(obj["messages"], list) and obj["messages"], f"line {i}: messages"
        for m in obj["messages"]:
            assert set(m) == {"role", "content"}, f"line {i}: message keys {set(m)}"
            assert m["role"] in ROLES, f"line {i}: role {m['role']!r}"
            assert isinstance(m["content"], str) and m["content"].strip(), f"line {i}: empty content"
        # last message must be the assistant answer we train on
        assert obj["messages"][-1]["role"] == "assistant", f"line {i}: last role"
        n_msgs += len(obj["messages"])
    print(f"OK: {len(ids)} examples, {n_msgs} messages total, ids unique")
    return 0


if __name__ == "__main__":
    sys.exit(main())
