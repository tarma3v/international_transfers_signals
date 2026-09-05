"""Verify that the frozen prospective model inputs remain byte-identical."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


MANIFEST = Path("results/research/round6/prospective_freeze/manifest.json")


def verify(manifest_path=MANIFEST):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    mismatches = []
    for raw_path, expected in manifest["sha256"].items():
        path = Path(raw_path)
        if not path.exists():
            mismatches.append({"path": raw_path, "status": "missing"})
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append({
                "path": raw_path,
                "status": "changed",
                "expected": expected,
                "actual": actual,
            })
    return mismatches


def main() -> None:
    mismatches = verify()
    if mismatches:
        raise SystemExit(json.dumps(mismatches, ensure_ascii=False, indent=2))
    print("prospective freeze verified")


if __name__ == "__main__":
    main()
