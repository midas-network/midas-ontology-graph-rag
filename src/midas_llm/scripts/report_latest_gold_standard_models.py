#!/usr/bin/env python3
"""Build a cross-model HTML report from the latest gold-standard runs.

This script scans `output/gold_standard/results`, picks the most recent run
subdirectory under each model directory, merges the model evaluation results,
and writes one easy-to-read HTML report for N-model comparison.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


@dataclass(frozen=True)
class RunSource:
    model_directory: str
    run_directory: Path
    run_timestamp: datetime
    evaluation_path: Path
    data: dict[str, Any]


def _parse_run_timestamp(name: str) -> datetime | None:
    try:
        return datetime.strptime(name, TIMESTAMP_FORMAT)
    except ValueError:
        return None


def _html_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _iter_model_directories(results_root: Path) -> list[Path]:
    return sorted(p for p in results_root.iterdir() if p.is_dir())


def find_latest_run_sources(results_root: Path) -> tuple[list[RunSource], list[str]]:
    """Return latest run per model directory and non-fatal warnings."""
    sources: list[RunSource] = []
    warnings: list[str] = []

    for model_dir in _iter_model_directories(results_root):
        run_candidates: list[tuple[datetime, Path]] = []
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_ts = _parse_run_timestamp(run_dir.name)
            if run_ts is None:
                continue
            run_candidates.append((run_ts, run_dir))

        if not run_candidates:
            warnings.append(
                f"{model_dir.name}: no timestamped run directories found."
            )
            continue

        run_candidates.sort(key=lambda item: (item[0], item[1].name))
        latest_ts, latest_run_dir = run_candidates[-1]
        evaluation_path = latest_run_dir / "evaluation.json"

        if not evaluation_path.is_file():
            warnings.append(
                f"{model_dir.name}: latest run ({latest_run_dir.name}) has no evaluation.json."
            )
            continue

        try:
            data = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(
                f"{model_dir.name}: invalid JSON in {evaluation_path}: {exc}"
            )
            continue

        sources.append(
            RunSource(
                model_directory=model_dir.name,
                run_directory=latest_run_dir,
                run_timestamp=latest_ts,
                evaluation_path=evaluation_path,
                data=data,
            )
        )

    return sources, warnings


def _build_combined_results(
    sources: list[RunSource],
) -> tuple[
    dict[str, Any],
    dict[str, RunSource],
    dict[str, Counter[str]],
    list[str],
]:
    """Merge latest-run results into one combined evaluation structure."""
    # Keep newest record if duplicate model+abstract appears across sources.
    model_abstract_best: dict[tuple[str, str], tuple[datetime, dict[str, Any]]] = {}
    abstract_titles: dict[str, str] = {}
    model_sources: dict[str, RunSource] = {}
    model_attribute_counters: dict[str, Counter[str]] = {}
    warnings: list[str] = []

    for source in sources:
        for abstract in source.data.get("abstracts", []):
            abstract_id = str(abstract.get("id", "unknown"))
            if abstract_id == "unknown":
                warnings.append(
                    f"{source.model_directory}/{source.run_directory.name}: abstract missing id."
                )
            title = str(abstract.get("title", "")).strip()
            if title and abstract_id not in abstract_titles:
                abstract_titles[abstract_id] = title

            models_obj = abstract.get("models", {})
            if not isinstance(models_obj, dict):
                continue

            for model_name, model_result in models_obj.items():
                if not isinstance(model_result, dict):
                    continue

                key = (str(model_name), abstract_id)
                incumbent = model_abstract_best.get(key)
                if incumbent is None or source.run_timestamp >= incumbent[0]:
                    model_abstract_best[key] = (source.run_timestamp, model_result)

                prev_source = model_sources.get(str(model_name))
                if prev_source is None or source.run_timestamp >= prev_source.run_timestamp:
                    model_sources[str(model_name)] = source

    abstracts_map: dict[str, dict[str, Any]] = {}
    for (model_name, abstract_id), (_, model_result) in model_abstract_best.items():
        abstract_entry = abstracts_map.setdefault(
            abstract_id,
            {
                "id": abstract_id,
                "title": abstract_titles.get(abstract_id, ""),
                "models": {},
            },
        )
        abstract_entry["models"][model_name] = model_result

        counters = model_attribute_counters.setdefault(
            model_name,
            Counter(),
        )
        evaluation = model_result.get("evaluation", {})
        if isinstance(evaluation, dict):
            for hit in evaluation.get("hits", []) or []:
                if isinstance(hit, dict):
                    attr = str(hit.get("attribute", ""))
                    if attr:
                        counters[f"hit::{attr}"] += 1
            for miss in evaluation.get("misses", []) or []:
                if isinstance(miss, dict):
                    attr = str(miss.get("attribute", ""))
                    if attr:
                        counters[f"miss::{attr}"] += 1
            for fp in evaluation.get("false_positives", []) or []:
                if isinstance(fp, dict):
                    attr = str(fp.get("attribute", ""))
                    if attr:
                        counters[f"fp::{attr}"] += 1

    abstracts = sorted(
        abstracts_map.values(),
        key=lambda a: (str(a.get("id", "")), str(a.get("title", ""))),
    )
    model_names = sorted(
        {
            model_name
            for abstract_payload in abstracts_map.values()
            for model_name in abstract_payload.get("models", {})
        }
    )

    combined_results = {
        "timestamp": datetime.now().isoformat(),
        "models": model_names,
        "abstracts": abstracts,
    }
    return combined_results, model_sources, model_attribute_counters, warnings


def _aggregate_model_scores(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    model_scores: dict[str, dict[str, Any]] = {}

    for abstract_result in results.get("abstracts", []):
        for model, model_result in abstract_result.get("models", {}).items():
            if not isinstance(model_result, dict) or "error" in model_result:
                continue

            bucket = model_scores.setdefault(
                model,
                {
                    "abstracts_evaluated": 0,
                    "total_expected": 0,
                    "total_hits": 0,
                    "total_misses": 0,
                    "total_false_positives": 0,
                    "request_duration_s_sum": 0.0,
                    "request_duration_s_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                },
            )

            scores = model_result.get("evaluation", {}).get("scores", {})
            bucket["abstracts_evaluated"] += 1
            bucket["total_expected"] += int(scores.get("total_expected", 0) or 0)
            bucket["total_hits"] += int(scores.get("total_hits", 0) or 0)
            bucket["total_misses"] += int(scores.get("total_misses", 0) or 0)
            bucket["total_false_positives"] += int(
                scores.get("total_false_positives", 0) or 0
            )

            timing = model_result.get("timing", {})
            if isinstance(timing, dict):
                req = timing.get("request_duration_s")
                if isinstance(req, (int, float)):
                    bucket["request_duration_s_sum"] += float(req)
                    bucket["request_duration_s_count"] += 1
                if isinstance(timing.get("prompt_tokens"), int):
                    bucket["prompt_tokens"] += timing["prompt_tokens"]
                if isinstance(timing.get("completion_tokens"), int):
                    bucket["completion_tokens"] += timing["completion_tokens"]
                if isinstance(timing.get("reasoning_tokens"), int):
                    bucket["reasoning_tokens"] += timing["reasoning_tokens"]

    for model, bucket in model_scores.items():
        total_expected = bucket["total_expected"]
        total_hits = bucket["total_hits"]
        total_fp = bucket["total_false_positives"]
        recall = total_hits / total_expected if total_expected > 0 else 0.0
        precision = total_hits / (total_hits + total_fp) if (total_hits + total_fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        bucket["recall"] = recall
        bucket["precision"] = precision
        bucket["f1"] = f1
        bucket["avg_request_duration_s"] = (
            bucket["request_duration_s_sum"] / bucket["request_duration_s_count"]
            if bucket["request_duration_s_count"] > 0
            else None
        )

    return model_scores


def _render_html(
    *,
    combined_results: dict[str, Any],
    model_scores: dict[str, dict[str, Any]],
    model_sources: dict[str, RunSource],
    selected_sources: list[RunSource],
    model_attribute_counters: dict[str, Counter[str]],
    warnings: list[str],
    output_path: Path,
    results_root: Path,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models_sorted = sorted(
        model_scores.keys(),
        key=lambda m: (
            -float(model_scores[m].get("f1", 0.0)),
            -float(model_scores[m].get("precision", 0.0)),
            -float(model_scores[m].get("recall", 0.0)),
            m.lower(),
        ),
    )

    html: list[str] = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='en'>")
    html.append("<head>")
    html.append("  <meta charset='UTF-8'>")
    html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html.append("  <title>Gold Standard Model Comparison Report</title>")
    html.append("  <style>")
    html.append("    :root { --bg:#f7f9fc; --card:#fff; --ink:#1f2933; --muted:#52606d; --line:#d9e2ec; --head:#f0f4f8; --good:#046c4e; --warn:#8d6c09; --bad:#b00020; }")
    html.append("    body { margin: 0; padding: 24px; font-family: Arial, sans-serif; background: var(--bg); color: var(--ink); }")
    html.append("    h1, h2, h3 { margin: 0 0 10px 0; }")
    html.append("    .sub { color: var(--muted); margin-bottom: 10px; }")
    html.append("    .card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 16px; margin-bottom: 18px; }")
    html.append("    table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 14px; }")
    html.append("    th, td { border: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }")
    html.append("    th { background: var(--head); font-weight: 600; position: sticky; top: 0; z-index: 1; }")
    html.append("    tr:nth-child(even) { background: #fbfcfe; }")
    html.append("    .metric-good { color: var(--good); font-weight: 600; }")
    html.append("    .metric-warn { color: var(--warn); font-weight: 600; }")
    html.append("    .metric-bad { color: var(--bad); font-weight: 600; }")
    html.append("    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; font-size: 12px; }")
    html.append("    .small { color: var(--muted); font-size: 12px; }")
    html.append("    details { margin-top: 8px; }")
    html.append("    summary { cursor: pointer; color: var(--muted); }")
    html.append("    .scroll { overflow-x: auto; }")
    html.append("  </style>")
    html.append("</head>")
    html.append("<body>")

    html.append("  <div class='card'>")
    html.append("    <h1>Gold Standard Cross-Model Report</h1>")
    html.append(f"    <div class='sub'>Generated: {_html_escape(generated_at)}</div>")
    html.append(f"    <div class='sub'>Results root: <span class='mono'>{_html_escape(results_root)}</span></div>")
    html.append(f"    <div class='sub'>Selected latest runs: {len(selected_sources)} model directories</div>")
    html.append(f"    <div class='sub'>Models compared: {len(models_sorted)}</div>")
    html.append(f"    <div class='sub'>Abstracts in merged view: {len(combined_results.get('abstracts', []))}</div>")
    html.append("  </div>")

    html.append("  <div class='card'>")
    html.append("    <h2>Selected Latest Run Per Model Directory</h2>")
    html.append("    <div class='scroll'><table>")
    html.append("      <thead><tr><th>Model Directory</th><th>Run Folder</th><th>Run Timestamp</th><th>Models Present in evaluation.json</th></tr></thead>")
    html.append("      <tbody>")
    for source in sorted(selected_sources, key=lambda s: (s.model_directory.lower(), s.run_timestamp)):
        models_present = source.data.get("models", [])
        models_display = ", ".join(str(m) for m in models_present) if isinstance(models_present, list) else ""
        html.append(
            "        <tr>"
            + f"<td>{_html_escape(source.model_directory)}</td>"
            + f"<td class='mono'>{_html_escape(source.run_directory)}</td>"
            + f"<td>{_html_escape(source.run_timestamp.strftime('%Y-%m-%d %H:%M:%S'))}</td>"
            + f"<td>{_html_escape(models_display)}</td>"
            + "</tr>"
        )
    html.append("      </tbody>")
    html.append("    </table></div>")
    html.append("  </div>")

    html.append("  <div class='card'>")
    html.append("    <h2>Model Leaderboard (Aggregated)</h2>")
    html.append("    <div class='small'>Scores are aggregated from each model's most recent available run data.</div>")
    html.append("    <div class='scroll'><table>")
    html.append("      <thead><tr><th>Rank</th><th>Model</th><th>Source Run</th><th>Abstracts</th><th>Expected</th><th>Hits</th><th>Misses</th><th>False Positives</th><th>Recall</th><th>Precision</th><th>F1</th><th>Avg Request</th><th>Prompt Tokens</th><th>Completion Tokens</th><th>Reasoning Tokens</th></tr></thead>")
    html.append("      <tbody>")
    for idx, model in enumerate(models_sorted, start=1):
        score = model_scores[model]
        source = model_sources.get(model)
        source_label = f"{source.model_directory}/{source.run_directory.name}" if source else "unknown"

        f1 = float(score.get("f1", 0.0))
        f1_class = "metric-good" if f1 >= 0.80 else ("metric-warn" if f1 >= 0.65 else "metric-bad")
        avg_req = score.get("avg_request_duration_s")
        avg_req_str = f"{avg_req:.2f}s" if isinstance(avg_req, (int, float)) else "n/a"

        html.append(
            "        <tr>"
            + f"<td>{idx}</td>"
            + f"<td>{_html_escape(model)}</td>"
            + f"<td class='mono'>{_html_escape(source_label)}</td>"
            + f"<td>{score['abstracts_evaluated']}</td>"
            + f"<td>{score['total_expected']}</td>"
            + f"<td>{score['total_hits']}</td>"
            + f"<td>{score['total_misses']}</td>"
            + f"<td>{score['total_false_positives']}</td>"
            + f"<td>{score['recall']:.2%}</td>"
            + f"<td>{score['precision']:.2%}</td>"
            + f"<td class='{f1_class}'>{score['f1']:.2%}</td>"
            + f"<td>{avg_req_str}</td>"
            + f"<td>{score['prompt_tokens']}</td>"
            + f"<td>{score['completion_tokens']}</td>"
            + f"<td>{score['reasoning_tokens']}</td>"
            + "</tr>"
        )
    html.append("      </tbody>")
    html.append("    </table></div>")
    html.append("  </div>")

    html.append("  <div class='card'>")
    html.append("    <h2>Per-Abstract Comparison Matrix</h2>")
    html.append("    <div class='small'>Cell format: F1, hits/misses/fp for that abstract and model.</div>")
    html.append("    <div class='scroll'><table>")
    html.append("      <thead><tr><th>Abstract</th>")
    for model in models_sorted:
        html.append(f"<th>{_html_escape(model)}</th>")
    html.append("</tr></thead>")
    html.append("      <tbody>")
    for abstract in combined_results.get("abstracts", []):
        abstract_id = str(abstract.get("id", "unknown"))
        title = str(abstract.get("title", "")).strip()
        label = f"{abstract_id} — {title}" if title else abstract_id
        html.append(f"        <tr><td>{_html_escape(label)}</td>")
        models_payload = abstract.get("models", {})
        for model in models_sorted:
            payload = models_payload.get(model)
            if not isinstance(payload, dict):
                html.append("<td class='small'>—</td>")
                continue
            if "error" in payload:
                html.append(f"<td class='metric-bad'>ERROR: {_html_escape(payload['error'])}</td>")
                continue
            score = payload.get("evaluation", {}).get("scores", {})
            f1 = float(score.get("f1", 0.0))
            hits = int(score.get("total_hits", 0) or 0)
            misses = int(score.get("total_misses", 0) or 0)
            fps = int(score.get("total_false_positives", 0) or 0)
            f1_class = "metric-good" if f1 >= 0.80 else ("metric-warn" if f1 >= 0.65 else "metric-bad")
            html.append(
                "<td>"
                + f"<div class='{f1_class}'>F1 {f1:.1%}</div>"
                + f"<div class='small'>h/m/fp: {hits}/{misses}/{fps}</div>"
                + "</td>"
            )
        html.append("</tr>")
    html.append("      </tbody>")
    html.append("    </table></div>")
    html.append("  </div>")

    html.append("  <div class='card'>")
    html.append("    <h2>Attribute-Level Signal (Top Misses and False Positives)</h2>")
    html.append("    <div class='small'>Counts are aggregated across all evaluated abstracts for each model.</div>")
    for model in models_sorted:
        counter = model_attribute_counters.get(model, Counter())
        miss_items = sorted(
            ((k.split("::", 1)[1], v) for k, v in counter.items() if k.startswith("miss::")),
            key=lambda x: (-x[1], x[0]),
        )[:10]
        fp_items = sorted(
            ((k.split("::", 1)[1], v) for k, v in counter.items() if k.startswith("fp::")),
            key=lambda x: (-x[1], x[0]),
        )[:10]

        html.append(f"    <details><summary>{_html_escape(model)}</summary>")
        html.append("      <div class='scroll'><table>")
        html.append("        <thead><tr><th>Top Miss Attributes</th><th>Count</th><th>Top False Positive Attributes</th><th>Count</th></tr></thead>")
        html.append("        <tbody>")
        row_count = max(len(miss_items), len(fp_items), 1)
        for i in range(row_count):
            miss_attr, miss_cnt = miss_items[i] if i < len(miss_items) else ("", "")
            fp_attr, fp_cnt = fp_items[i] if i < len(fp_items) else ("", "")
            html.append(
                "          <tr>"
                + f"<td>{_html_escape(miss_attr)}</td><td>{_html_escape(miss_cnt)}</td>"
                + f"<td>{_html_escape(fp_attr)}</td><td>{_html_escape(fp_cnt)}</td>"
                + "</tr>"
            )
        html.append("        </tbody>")
        html.append("      </table></div>")
        html.append("    </details>")
    html.append("  </div>")

    if warnings:
        html.append("  <div class='card'>")
        html.append("    <h2>Warnings</h2>")
        html.append("    <ul>")
        for warning in warnings:
            html.append(f"      <li>{_html_escape(warning)}</li>")
        html.append("    </ul>")
        html.append("  </div>")

    html.append("</body>")
    html.append("</html>")

    output_path.write_text("\n".join(html), encoding="utf-8")
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a cross-model HTML report by taking the latest run directory "
            "under each model directory in output/gold_standard/results."
        )
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="output/gold_standard/results",
        help="Root directory containing per-model run folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output HTML file path. Default: "
            "<results-root>/latest_models_report_<timestamp>.html"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()

    if not results_root.is_dir():
        raise SystemExit(f"Results root does not exist or is not a directory: {results_root}")

    now_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = (
        Path(args.output).resolve()
        if args.output
        else results_root / f"latest_models_report_{now_tag}.html"
    )

    sources, source_warnings = find_latest_run_sources(results_root)
    if not sources:
        raise SystemExit(
            "No valid latest runs with evaluation.json were found under: "
            f"{results_root}"
        )

    combined_results, model_sources, attribute_counters, merge_warnings = _build_combined_results(
        sources
    )
    model_scores = _aggregate_model_scores(combined_results)

    report_path = _render_html(
        combined_results=combined_results,
        model_scores=model_scores,
        model_sources=model_sources,
        selected_sources=sources,
        model_attribute_counters=attribute_counters,
        warnings=source_warnings + merge_warnings,
        output_path=output_path,
        results_root=results_root,
    )

    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
