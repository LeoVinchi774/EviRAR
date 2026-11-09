#!/usr/bin/env python3
"""Recalculate ATCMD-Bench score means from a per-case JSONL result file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evirar.scoring import SCORE_FIELDS, build_score_summary  # noqa: E402


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read result file {path}: {exc}") from exc

    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Result line {line_number} must contain a JSON object")
        records.append(record)
    if not records:
        raise ValueError(f"Result file contains no records: {path}")
    return records


def summarize_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    records = load_records(input_path)
    summary = build_score_summary(records)
    if summary["scored_cases"] == 0:
        fields = ", ".join(SCORE_FIELDS)
        raise ValueError(f"No complete expert scores found; expected fields: {fields}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize ATCMD-Bench expert scores from an EviRAR JSONL result file."
    )
    parser.add_argument("result_jsonl", type=Path, help="Per-case results/<mode>_<UTC time>.jsonl")
    parser.add_argument(
        "--output", type=Path,
        help="Summary JSON path (default: <input stem>_summary.json beside the input)",
    )
    args = parser.parse_args(argv)
    input_path = args.result_jsonl.expanduser()
    output_path = args.output.expanduser() if args.output else input_path.with_name(
        f"{input_path.stem}_summary.json"
    )
    try:
        summary = summarize_file(input_path, output_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Cases: {summary['cases']}; scored: {summary['scored_cases']}")
    for field, score in summary["means"].items():
        print(f"{field}: {score:.2f}/100")
    print(f"Summary written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
