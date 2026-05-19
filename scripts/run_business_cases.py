from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = {
    "retail_standard_refund": "scenarios/retail/scen_020_standard_refund.json",
    "retail_fraud_disclosure": "scenarios/retail/scen_027_fraud_disclosure.json",
    "helpdesk_cross_employee_disclosure": "scenarios/helpdesk/scen_046_cross_employee_disclosure.json",
    "finra_privacy_shield_trust_wire": "scenarios/finra/scen_059_privacy_shield_trust_wire.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run selected real business Pi-Bench cases and write a presentation report."
    )
    parser.add_argument("--pi-bench-root", default="../pi-bench", help="Path to the Pi-Bench repo")
    parser.add_argument("--host", default="127.0.0.1", help="Running agent host")
    parser.add_argument("--port", type=int, default=9010, help="Running agent port")
    parser.add_argument(
        "--cases",
        default=",".join(DEFAULT_CASES),
        help="Comma-separated case aliases, scenario IDs, stems, or JSON paths",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user-port", type=int, default=8768)
    parser.add_argument("--output-dir", default="generated/business_cases")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable or launcher command with Pi-Bench dependencies installed",
    )
    args = parser.parse_args()

    pi_bench_root = Path(args.pi_bench_root).resolve()
    if not (pi_bench_root / "examples" / "a2a_demo" / "run_a2a.py").exists():
        print(f"Pi-Bench runner not found under {pi_bench_root}", file=sys.stderr)
        return 2

    output_dir = (ROOT / args.output_dir).resolve()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / run_id
    selected_dir = run_dir / "selected_scenarios"
    selected_dir.mkdir(parents=True, exist_ok=True)

    selected = resolve_cases(pi_bench_root, args.cases)
    scenario_meta = {}
    for path in selected:
        data = json.loads(path.read_text(encoding="utf-8"))
        scenario_id = data.get("meta", {}).get("scenario_id", path.stem)
        scenario_meta[scenario_id] = data
        domain = _domain_folder(data, path)
        target_dir = selected_dir / domain
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_dir / path.name)

    raw_path = run_dir / "pibench-results.json"
    report_path = run_dir / "business-case-report.md"
    summary_path = run_dir / "business-case-summary.json"

    cmd = [
        *_python_command(args.python),
        "examples/a2a_demo/run_a2a.py",
        "--external",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--serve-user",
        "--user-kind",
        "scripted",
        "--user-port",
        str(args.user_port),
        "--scenarios-dir",
        str(selected_dir),
        "--concurrency",
        "1",
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(args.seed),
        "--save-to",
        str(raw_path),
    ]
    env = os.environ.copy()
    pi_bench_src = str(pi_bench_root / "src")
    env["PYTHONPATH"] = (
        pi_bench_src
        if not env.get("PYTHONPATH")
        else pi_bench_src + os.pathsep + env["PYTHONPATH"]
    )
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(cmd, cwd=pi_bench_root, env=env)

    if not raw_path.exists():
        print(f"Pi-Bench did not write {raw_path}", file=sys.stderr)
        return completed.returncode or 2

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    summary = summarize_results(results, scenario_meta)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_markdown(summary, payload), encoding="utf-8")

    print(f"report={report_path}")
    print(f"summary={summary_path}")
    print(f"raw={raw_path}")
    return completed.returncode


def resolve_cases(pi_bench_root: Path, cases: str) -> list[Path]:
    tokens = [token.strip() for token in cases.split(",") if token.strip()]
    if not tokens:
        tokens = list(DEFAULT_CASES)

    index = _scenario_index(pi_bench_root)
    resolved: list[Path] = []
    for token in tokens:
        path_text = DEFAULT_CASES.get(token, token)
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = pi_bench_root / candidate
        if candidate.exists():
            resolved.append(candidate)
            continue
        key = token.upper()
        if key in index:
            resolved.append(index[key])
            continue
        if token in index:
            resolved.append(index[token])
            continue
        raise SystemExit(f"Unknown business case: {token}")
    return resolved


def _scenario_index(pi_bench_root: Path) -> dict[str, Path]:
    scenarios_dir = pi_bench_root / "scenarios"
    index: dict[str, Path] = {}
    for path in scenarios_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("schema_version") != "pibench_scenario_v1":
            continue
        scenario_id = str(data.get("meta", {}).get("scenario_id", ""))
        if scenario_id:
            index[scenario_id] = path
        index[path.stem] = path
    return index


def summarize_results(results: list[dict[str, Any]], scenario_meta: dict[str, dict]) -> dict[str, Any]:
    cases = []
    for result in results:
        scenario_id = str(result.get("scenario_id", ""))
        data = scenario_meta.get(scenario_id, {})
        meta = data.get("meta", {})
        leaderboard = data.get("leaderboard", {})
        failed = [
            outcome
            for outcome in result.get("outcome_results", [])
            if not outcome.get("passed", False)
        ]
        cases.append(
            {
                "scenario_id": scenario_id,
                "title": _title_from_id(scenario_id),
                "domain": result.get("domain_name") or result.get("domain"),
                "business_risk": meta.get("notes", leaderboard.get("primary", "")),
                "leaderboard_primary": leaderboard.get("primary", result.get("leaderboard_primary", "")),
                "expected_decision": result.get("label"),
                "actual_decision": result.get("canonical_decision"),
                "decision_error": result.get("decision_error"),
                "passed": bool(result.get("all_passed")),
                "reward": result.get("reward"),
                "duration_seconds": result.get("duration"),
                "tool_sequence": _tool_sequence(result),
                "failure_category": classify_failure(result),
                "failed_outcomes": [
                    {
                        "outcome_id": item.get("outcome_id"),
                        "type": item.get("type"),
                        "detail": item.get("detail"),
                    }
                    for item in failed[:8]
                ],
                "event_flags": result.get("event_flags", {}),
            }
        )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(cases),
        "passed": sum(1 for item in cases if item["passed"]),
        "cases": cases,
    }


def classify_failure(result: dict[str, Any]) -> str:
    if result.get("status") == "error":
        error = str(result.get("error", "")).lower()
        if any(term in error for term in ("http", "a2a", "json-rpc", "connect")):
            return "protocol"
        return "protocol"

    decision_error = str(result.get("decision_error", "") or "")
    if decision_error and decision_error not in {"NONE", "None"}:
        if "MISSING" in decision_error:
            return "parsing"
        if "INVALID" in decision_error:
            return "invalid_tool_call"

    failed = [
        outcome
        for outcome in result.get("outcome_results", [])
        if not outcome.get("passed", False)
    ]
    types = {str(item.get("type", "")) for item in failed}
    details = " ".join(str(item.get("detail", "")) for item in failed).lower()
    if any(t in types for t in ("tool_called_with", "tool_arg_equals")):
        return "tool_argument"
    if any(t in types for t in ("tool_before_tool", "state_field")):
        return "ordering/state"
    if "decision_equals" in types:
        return "wrong_decision"
    if result.get("event_flags", {}).get("UR_r"):
        return "under-refusal"
    if result.get("event_flags", {}).get("OR_r"):
        return "over-refusal"
    if "tool_called(" in details and "=false" in details:
        return "invalid_tool_call"
    return "passed" if result.get("all_passed") else "wrong_decision"


def render_markdown(summary: dict[str, Any], payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        "# CoreLink Policy Agent Business Case Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Cases: {summary['passed']}/{summary['total']} passed",
        "",
    ]
    if metrics:
        lines.extend(
            [
                "## Benchmark Summary",
                "",
                f"- Overall score: {metrics.get('overall_score', 'n/a')}",
                f"- Compliance rate: {metrics.get('compliance_rate', 'n/a')}",
                f"- Errors: {metrics.get('errors', 'n/a')}",
                "",
            ]
        )
    for case in summary["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        lines.extend(
            [
                f"## {case['title']} ({status})",
                "",
                f"- Scenario: `{case['scenario_id']}`",
                f"- Domain: `{case['domain']}`",
                f"- Business risk: {case['business_risk']}",
                f"- Evaluation focus: {case['leaderboard_primary']}",
                f"- Expected decision: `{case['expected_decision']}`",
                f"- Actual decision: `{case['actual_decision']}`",
                f"- Failure category: `{case['failure_category']}`",
                f"- Tool sequence: `{', '.join(case['tool_sequence']) or 'none'}`",
                f"- Event flags: `{json.dumps(case['event_flags'], sort_keys=True)}`",
                "",
            ]
        )
        if case["failed_outcomes"]:
            lines.append("Failed checks:")
            for outcome in case["failed_outcomes"]:
                lines.append(
                    f"- `{outcome['outcome_id']}` ({outcome['type']}): {outcome['detail']}"
                )
            lines.append("")
    return "\n".join(lines)


def _tool_sequence(result: dict[str, Any]) -> list[str]:
    sequence: list[str] = []
    for call in result.get("tool_calls", []) or []:
        if isinstance(call, str):
            sequence.append(call)
            continue
        if isinstance(call, dict):
            name = call.get("tool_name")
            if not name and isinstance(call.get("function"), dict):
                name = call["function"].get("name")
            if name:
                sequence.append(str(name))
    return sequence


def _domain_folder(data: dict[str, Any], path: Path) -> str:
    domain = str(data.get("meta", {}).get("domain", "")).lower()
    if "helpdesk" in domain:
        return "helpdesk"
    if "retail" in domain:
        return "retail"
    if "finra" in domain:
        return "finra"
    return path.parent.name


def _title_from_id(scenario_id: str) -> str:
    return scenario_id.replace("SCEN_", "").replace("_", " ").title()


def _python_command(value: str) -> list[str]:
    return shlex.split(value, posix=os.name != "nt") or [sys.executable]


if __name__ == "__main__":
    raise SystemExit(main())
