"""Command line interface for EviRAR consultations."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from evirar.benchmark import CaseSimulator, load_dataset
from evirar.clients import Runtime
from evirar.config import load_settings
from evirar.consultation import Consultation, simulated_patient
from evirar.scoring import build_score_summary, score_dialogue


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evirar", description="Run a TCM consultation with Baseline or EviRAR.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--env-file", help="Optional .env file path")
    case_source = parser.add_mutually_exclusive_group()
    case_source.add_argument("--complaint", help="Initial complaint for a single consultation")
    case_source.add_argument("--dataset", help="Run cases from an ATCMD-Bench JSON dataset")
    parser.add_argument("--simulated-patient", action="store_true", help="Use a model as the patient/assistant instead of typing replies")
    parser.add_argument("--limit", type=int, help="Maximum dataset cases to run (default: all cases)")
    parser.add_argument("--output-jsonl", help="Dataset result path (default: results/<mode>_<UTC timestamp>.jsonl)")
    parser.add_argument("--check", action="store_true", help="Validate settings and bundled resource cards without calling APIs")
    parser.add_argument("--verbose", action="store_true", help="Show module logs, including clinical and retrieval details")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args.config, env_file=args.env_file)
        if args.check:
            counts = settings.validate_resources()
            print(f"Configuration valid: mode={settings.values['mode']}; resources={counts or 'not required for Baseline'}")
            return 0
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be a positive integer")
        if args.limit is not None and not args.dataset:
            raise ValueError("--limit can only be used with --dataset")
        settings.validate_services(simulator=args.simulated_patient)
        if args.dataset:
            if not args.simulated_patient:
                raise ValueError("ATCMD-Bench dataset runs require --simulated-patient")
            cases = load_dataset(args.dataset)
            if args.limit is not None:
                cases = cases[:args.limit]
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = Path(args.output_jsonl or f"results/{settings.values['mode']}_{timestamp}.jsonl").expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            score_records = []
            with output_path.open("x", encoding="utf-8") as output:
                for index, case in enumerate(cases, start=1):
                    # Context modules contain clinical state; none may leak
                    # from an earlier case into the next consultation.
                    consultation = Consultation(settings)
                    try:
                        with consultation.runtime.activate():
                            simulator = CaseSimulator(consultation.runtime, case)
                            result = consultation.run(
                                simulator.opening(), simulator,
                                lambda reply: print(f"\n医生：{reply}"),
                            )
                            scores = score_dialogue(consultation.runtime, case, consultation.messages[1:])
                            consultation.runtime.ensure_success()
                        record = {
                            "id": case["id"],
                            "department": case["科室"],
                            "mode": settings.values["mode"],
                            "diagnosis": result,
                            "dialogue": consultation.messages[1:],
                            **scores,
                            "self_reflection": consultation.reflection_result,
                            "kb_gate_turns": consultation.context._kb_gate_turns,
                            "api_events": consultation.runtime.events,
                        }
                        score_records.append(record)
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                        print(f"\n已完成病例 {index}/{len(cases)}，ID={case['id']}，结果写入 {output_path}")
                    finally:
                        consultation.close()
            summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
            summary_path.write_text(
                json.dumps(build_score_summary(score_records), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"\n数据集运行完成：{len(cases)} 例；结果文件：{output_path}；评分汇总：{summary_path}")
            return 0
        complaint = args.complaint if args.complaint is not None else input("患者主诉：").strip()
        consultation = Consultation(settings)
        try:
            with consultation.runtime.activate():
                patient = simulated_patient(consultation.runtime) if args.simulated_patient else lambda _history: input("患者/助理回复：")
                print("医生：")
                result = consultation.run(complaint, patient, lambda reply: print(f"\n医生：{reply}"))
                consultation.runtime.ensure_success()
                print("\n本次问诊结束。API 调用统计：")
                for kind in sorted({event["service"] for event in consultation.runtime.events}):
                    events = [event for event in consultation.runtime.events if event["service"] == kind]
                    print(f"  {kind}: {len(events)} 次，{sum(e['elapsed_seconds'] for e in events):.1f}s")
            return 0 if result else 1
        finally:
            consultation.close()
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
