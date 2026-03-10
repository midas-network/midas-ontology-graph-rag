"""Generate human-readable evaluation reports for resources scientists.

This module produces text reports showing hits, misses, and false positives
from LLM extraction evaluation against a gold standard.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from concept_extractor.constants import DEFAULT_EXTRACTION_OUTPUT_DIR

LOGGER = logging.getLogger("midas-llm")


def _vector_model_score_sort_key(entry: dict[str, Any], original_index: int) -> tuple[int, float, int, str, int]:
    """Sort key for vector model score rows (highest similarity first)."""
    score = entry.get("similarity_score")
    numeric_score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None

    # Keep deterministic ordering when scores tie.
    decision_rank = {
        "MATCH": 0,
        "AMBIGUOUS": 1,
        "NO_MATCH": 2,
        "UNAVAILABLE": 3,
    }.get(str(entry.get("decision", "unknown")).upper(), 99)

    return (
        0 if numeric_score is not None else 1,
        -(numeric_score if numeric_score is not None else 0.0),
        decision_rank,
        str(entry.get("model", "unknown")).lower(),
        original_index,
    )


def _format_vector_model_scores(scores: Any) -> str:
    """Format per-model vector similarity decisions for report display."""
    if not isinstance(scores, list) or not scores:
        return ""

    valid_entries: list[tuple[int, dict[str, Any]]] = [
        (idx, entry) for idx, entry in enumerate(scores) if isinstance(entry, dict)
    ]
    valid_entries.sort(key=lambda item: _vector_model_score_sort_key(item[1], item[0]))

    parts: list[str] = []
    for _, entry in valid_entries:
        model = str(entry.get("model", "unknown"))
        decision = str(entry.get("decision", "unknown"))
        score = entry.get("similarity_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            part = f"{model}:{decision}:{score:.3f}"
        else:
            part = f"{model}:{decision}"
        best_match = entry.get("best_match")
        if best_match:
            part = f"{part}:{best_match}"
        parts.append(part)
    return "; ".join(parts)


def generate_evaluation_text_report(
    results: dict[str, Any],
    output_dir: str = DEFAULT_EXTRACTION_OUTPUT_DIR,
    run_folder: str | None = None,
    logger: logging.Logger = LOGGER,
) -> str:
    """Generate a human-readable text report of evaluation results.

    Args:
        results: Evaluation results dict containing abstracts and model evaluations
        output_dir: Directory to save the report (used if run_folder not provided)
        run_folder: Specific run folder to save report in (takes precedence)
        logger: Logger instance

    Returns:
        Path to the generated report file
    """
    if run_folder:
        os.makedirs(run_folder, exist_ok=True)
        output_file = os.path.join(run_folder, "evaluation_report.txt")
    else:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")

    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("LLM EXTRACTION EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Models Evaluated: {', '.join(results.get('models', ['unknown']))}")
    lines.append(f"Abstracts Evaluated: {len(results.get('abstracts', []))}")
    lines.append("")

    # Evaluation configuration (if present)
    eval_config = results.get("evaluation_config", {})
    if eval_config:
        lines.append("-" * 80)
        lines.append("EVALUATION CONFIGURATION")
        lines.append("-" * 80)
        lines.append("")
        lines.append(f"  Vector Similarity:  {'ENABLED' if eval_config.get('use_vector_eval', False) else 'DISABLED'}")
        if eval_config.get("use_vector_eval", False):
            embedding_models = eval_config.get("embedding_models")
            if isinstance(embedding_models, list) and embedding_models:
                lines.append(f"    Embedding Models:   {', '.join(str(m) for m in embedding_models)}")
            else:
                lines.append(f"    Embedding Model:    {eval_config.get('embedding_model', 'unknown')}")
            lines.append(f"    Auto-match threshold: >= {eval_config.get('vector_high_threshold', 0.85):.2f}")
            lines.append(f"    Auto-reject threshold: <= {eval_config.get('vector_low_threshold', 0.50):.2f}")
        lines.append(f"  LLM Semantic Eval:  {'ENABLED' if eval_config.get('use_llm_eval', False) else 'DISABLED'}")
        lines.append("")

    # Legend
    lines.append("-" * 80)
    lines.append("EVALUATION LEGEND")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  ✓ HIT           The LLM extracted a value that semantically matches")
    lines.append("                  something in the gold standard.")
    lines.append("")
    lines.append("  ✗ MISS          The gold standard expects a value, but the LLM")
    lines.append("                  didn't extract it. This indicates the LLM failed")
    lines.append("                  to identify an expected attribute from the abstract.")
    lines.append("")
    lines.append("  ? FALSE POSITIVE The LLM extracted a value that is NOT in the gold")
    lines.append("                  standard. This could mean:")
    lines.append("                  - The LLM hallucinated information not in the text")
    lines.append("                  - The LLM interpreted something differently")
    lines.append("                  - The gold standard may be incomplete")
    lines.append("")
    lines.append("-" * 80)
    lines.append("")

    # Process each abstract
    for abstract_result in results.get("abstracts", []):
        abstract_id = abstract_result.get("id", "unknown")
        title = abstract_result.get("title", "Unknown Title")

        lines.append("=" * 80)
        lines.append(f"ABSTRACT: {abstract_id}")
        lines.append(f"TITLE: {title}")
        lines.append("=" * 80)
        lines.append("")

        # Process each model's results for this abstract
        for model, model_result in abstract_result.get("models", {}).items():
            if "error" in model_result:
                lines.append(f"  Model {model}: ERROR - {model_result['error']}")
                lines.append("")
                continue

            evaluation = model_result.get("evaluation", {})
            scores = evaluation.get("scores", {})

            # Model summary header
            lines.append("-" * 60)
            lines.append(f"MODEL: {model}")
            lines.append("-" * 60)
            lines.append("")

            # Scores summary
            recall = scores.get("recall", 0)
            precision = scores.get("precision", 0)
            f1 = scores.get("f1", 0)
            hits = scores.get("total_hits", 0)
            misses = scores.get("total_misses", 0)
            fps = scores.get("total_false_positives", 0)

            lines.append("PERFORMANCE METRICS:")
            lines.append(f"  Recall:    {recall:.2%}  (How many expected values were found)")
            lines.append(f"  Precision: {precision:.2%}  (How many extracted values were correct)")
            lines.append(f"  F1 Score:  {f1:.2%}  (Harmonic mean of recall and precision)")
            lines.append("")
            lines.append(f"  Total Hits:            {hits:3d}")
            lines.append(f"  Total Misses:          {misses:3d}")
            lines.append(f"  Total False Positives: {fps:3d}")
            lines.append("")

            # Vector evaluation stats
            vector_stats = evaluation.get("vector_stats", {})
            if any(vector_stats.values()):
                lines.append("VECTOR EVALUATION STATS:")
                lines.append(f"  Auto-matches (high confidence):  {vector_stats.get('auto_matches', 0):3d}")
                lines.append(f"  Auto-rejects (low confidence):   {vector_stats.get('auto_rejects', 0):3d}")
                lines.append(f"  Ambiguous → LLM fallback:        {vector_stats.get('ambiguous_to_llm', 0):3d}")
                if vector_stats.get("vector_unavailable", 0) > 0:
                    lines.append(f"  Vector unavailable:              {vector_stats.get('vector_unavailable', 0):3d}")
                lines.append("")

            # HITS section
            hit_list = evaluation.get("hits", [])
            lines.append(f"HITS ({len(hit_list)})  ─────────────────────────────────────────────")
            lines.append("  Values the LLM correctly extracted:")
            lines.append("")

            if hit_list:
                # Group hits by attribute
                hits_by_attr = _group_by_attribute(hit_list, "attribute")
                for attr, items in sorted(hits_by_attr.items()):
                    for item in items:
                        extracted = item.get("extracted_value", "")
                        matched = item.get("matched_expected", "")
                        method = item.get("match_method", "unknown")
                        similarity = item.get("similarity_score")

                        lines.append(f"  ✓ {attr}")
                        lines.append(f"      LLM extracted: \"{extracted}\"")
                        lines.append(f"      Matched gold:  \"{matched}\"")
                        lines.append(f"      Match method:  {method}")
                        if similarity is not None:
                            lines.append(f"      Similarity:    {similarity:.3f}")
                        vector_model = item.get("vector_selected_model")
                        if vector_model:
                            lines.append(f"      Vector model used: {vector_model}")
                        vector_details = _format_vector_model_scores(item.get("vector_model_scores"))
                        if vector_details:
                            lines.append(f"      Vector details:   {vector_details}")
                        lines.append("")
            else:
                lines.append("  (No hits)")
                lines.append("")

            # MISSES section
            miss_list = evaluation.get("misses", [])
            lines.append(f"MISSES ({len(miss_list)})  ───────────────────────────────────────────")
            lines.append("  Expected values the LLM failed to extract:")
            lines.append("")

            if miss_list:
                # Group misses by attribute
                misses_by_attr = _group_by_attribute(miss_list, "attribute")
                for attr, items in sorted(misses_by_attr.items()):
                    for item in items:
                        expected = item.get("expected_value", "")
                        lines.append(f"  ✗ {attr}")
                        lines.append(f"      Expected: \"{expected}\"")
                        lines.append(f"      Why missed: The LLM did not identify this value from the abstract.")
                        lines.append(f"                  Check if the abstract contains this information.")
                        lines.append("")
            else:
                lines.append("  (No misses - all expected values were found!)")
                lines.append("")

            # FALSE POSITIVES section
            fp_list = evaluation.get("false_positives", [])
            lines.append(f"FALSE POSITIVES ({len(fp_list)})  ─────────────────────────────────────")
            lines.append("  Values the LLM extracted that are NOT in the gold standard:")
            lines.append("")

            if fp_list:
                # Group false positives by attribute
                fps_by_attr = _group_by_attribute(fp_list, "attribute")
                for attr, items in sorted(fps_by_attr.items()):
                    for item in items:
                        extracted = item.get("extracted_value", "")
                        expected_vals = item.get("expected_values", [])
                        expected_str = ", ".join(f'"{v}"' for v in expected_vals)
                        similarity = item.get("similarity_score")

                        lines.append(f"  ? {attr}")
                        lines.append(f"      LLM extracted: \"{extracted}\"")
                        lines.append(f"      Expected values: [{expected_str}]")
                        if similarity is not None:
                            lines.append(f"      Best similarity: {similarity:.3f}")
                        vector_model = item.get("vector_selected_model")
                        if vector_model:
                            lines.append(f"      Vector model used: {vector_model}")
                        vector_details = _format_vector_model_scores(item.get("vector_model_scores"))
                        if vector_details:
                            lines.append(f"      Vector details:   {vector_details}")
                        lines.append(f"      Possible causes:")
                        lines.append(f"        - LLM may have hallucinated or misinterpreted")
                        lines.append(f"        - Value might be valid but missing from gold standard")
                        lines.append(f"        - Semantic match algorithm may have missed a valid match")
                        lines.append("")
            else:
                lines.append("  (No false positives - all extracted values were correct!)")
                lines.append("")

            lines.append("")

    # Overall Summary
    lines.append("=" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    model_scores = _aggregate_model_scores(results)

    for model, scores in model_scores.items():
        total_expected = scores["total_expected"]
        total_hits = scores["total_hits"]
        total_fps = scores["total_false_positives"]

        recall = total_hits / total_expected if total_expected > 0 else 0
        precision = total_hits / (total_hits + total_fps) if (total_hits + total_fps) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        lines.append(f"Model: {model}")
        lines.append(f"  Abstracts Evaluated: {scores['abstracts_evaluated']}")
        lines.append(f"  Total Expected Values: {total_expected}")
        lines.append(f"  Hits: {total_hits}")
        lines.append(f"  Misses: {scores['total_misses']}")
        lines.append(f"  False Positives: {total_fps}")
        lines.append(f"  Recall: {recall:.2%}")
        lines.append(f"  Precision: {precision:.2%}")
        lines.append(f"  F1 Score: {f1:.2%}")
        lines.append("")

    # Recommendations
    lines.append("-" * 80)
    lines.append("RECOMMENDATIONS FOR IMPROVEMENT")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Based on the evaluation results, consider:")
    lines.append("")
    lines.append("  1. For MISSES (low recall):")
    lines.append("     - Review if the abstract actually contains the expected information")
    lines.append("     - Improve prompt instructions for commonly missed attributes")
    lines.append("     - Add few-shot examples demonstrating difficult extractions")
    lines.append("")
    lines.append("  2. For FALSE POSITIVES (low precision):")
    lines.append("     - Add prompt instructions to avoid over-extraction")
    lines.append("     - Review if gold standard needs updating with valid values")
    lines.append("     - Check if LLM is inferring values not explicitly stated")
    lines.append("")
    lines.append("  3. For improving both:")
    lines.append("     - Test with different LLM models")
    lines.append("     - Adjust prompt complexity (simple vs detailed)")
    lines.append("     - Review and expand training examples")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    # Write to file
    report_content = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("Evaluation report saved to: %s", output_file)
    return output_file


def generate_evaluation_html_report(
    results: dict[str, Any],
    output_dir: str = DEFAULT_EXTRACTION_OUTPUT_DIR,
    run_folder: str | None = None,
    logger: logging.Logger = LOGGER,
) -> str:
    """Generate an HTML evaluation report with tabular hits/misses by model."""
    if run_folder:
        os.makedirs(run_folder, exist_ok=True)
        output_file = os.path.join(run_folder, "evaluation_report.html")
    else:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.html")

    model_scores = _aggregate_model_scores(results)
    model_events = _collect_model_events(results)

    html_parts: list[str] = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html lang='en'>")
    html_parts.append("<head>")
    html_parts.append("  <meta charset='UTF-8'>")
    html_parts.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html_parts.append("  <title>LLM Extraction Evaluation Report</title>")
    html_parts.append("  <style>")
    html_parts.append("    body { font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }")
    html_parts.append("    h1, h2 { margin-bottom: 8px; }")
    html_parts.append("    .meta { color: #52606d; margin-bottom: 16px; }")
    html_parts.append("    table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }")
    html_parts.append("    th, td { border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }")
    html_parts.append("    th { background: #f0f4f8; font-weight: 600; }")
    html_parts.append("    tr:nth-child(even) { background: #f8fafc; }")
    html_parts.append("    .tag { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: 600; }")
    html_parts.append("    .hit { background: #e3f9e5; color: #046c4e; border: 1px solid #57ae5b; }")
    html_parts.append("    .miss { background: #ffe3e3; color: #b00020; border: 1px solid #f8b4b4; }")
    html_parts.append("    .fp { background: #fff3c4; color: #8d6c09; border: 1px solid #f2d024; }")
    html_parts.append("    .section { margin-top: 32px; }")
    html_parts.append("    .small { color: #52606d; font-size: 13px; }")
    html_parts.append("  </style>")
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("  <h1>LLM Extraction Evaluation Report</h1>")
    html_parts.append("  <div class='meta'>Generated: " + _html_escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "</div>")
    html_parts.append("  <div class='meta'>Models Evaluated: " + _html_escape(', '.join(results.get('models', ['unknown']))) + "</div>")
    html_parts.append("  <div class='meta'>Abstracts Evaluated: " + str(len(results.get('abstracts', []))) + "</div>")

    eval_config = results.get("evaluation_config", {})
    if eval_config:
        html_parts.append("  <div class='section'>")
        html_parts.append("    <h2>Evaluation Configuration</h2>")
        html_parts.append("    <table>")
        html_parts.append("      <tbody>")
        html_parts.append("        <tr><th>Vector Similarity</th><td>" + ("ENABLED" if eval_config.get("use_vector_eval", False) else "DISABLED") + "</td></tr>")
        if eval_config.get("use_vector_eval", False):
            embedding_models = eval_config.get("embedding_models")
            if isinstance(embedding_models, list) and embedding_models:
                html_parts.append("        <tr><th>Embedding Models</th><td>" + _html_escape(", ".join(str(m) for m in embedding_models)) + "</td></tr>")
            else:
                html_parts.append("        <tr><th>Embedding Model</th><td>" + _html_escape(str(eval_config.get("embedding_model", "unknown"))) + "</td></tr>")
            html_parts.append("        <tr><th>Auto-match Threshold</th><td>&gt;= " + f"{eval_config.get('vector_high_threshold', 0.85):.2f}" + "</td></tr>")
            html_parts.append("        <tr><th>Auto-reject Threshold</th><td>&lt;= " + f"{eval_config.get('vector_low_threshold', 0.50):.2f}" + "</td></tr>")
        html_parts.append("        <tr><th>LLM Semantic Eval</th><td>" + ("ENABLED" if eval_config.get("use_llm_eval", False) else "DISABLED") + "</td></tr>")
        html_parts.append("      </tbody>")
        html_parts.append("    </table>")
        html_parts.append("  </div>")

    html_parts.append("  <div class='section'>")
    html_parts.append("    <h2>Model Summary</h2>")
    html_parts.append("    <table>")
    html_parts.append("      <thead><tr><th>Model</th><th>Abstracts</th><th>Expected</th><th>Hits</th><th>Misses</th><th>False Positives</th><th>Recall</th><th>Precision</th><th>F1</th></tr></thead>")
    html_parts.append("      <tbody>")
    for model, scores in model_scores.items():
        total_expected = scores["total_expected"] or 0
        total_hits = scores["total_hits"] or 0
        total_fps = scores["total_false_positives"] or 0
        recall = total_hits / total_expected if total_expected else 0
        precision = total_hits / (total_hits + total_fps) if (total_hits + total_fps) else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        html_parts.append("        <tr>" +
                          "<td>" + _html_escape(model) + "</td>" +
                          "<td>" + str(scores["abstracts_evaluated"]) + "</td>" +
                          "<td>" + str(total_expected) + "</td>" +
                          "<td>" + str(total_hits) + "</td>" +
                          "<td>" + str(scores["total_misses"]) + "</td>" +
                          "<td>" + str(total_fps) + "</td>" +
                          "<td>" + f"{recall:.2%}" + "</td>" +
                          "<td>" + f"{precision:.2%}" + "</td>" +
                          "<td>" + f"{f1:.2%}" + "</td>" +
                          "</tr>")
    html_parts.append("      </tbody>")
    html_parts.append("    </table>")
    html_parts.append("  </div>")

    for model, rows in model_events.items():
        html_parts.append("  <div class='section'>")
        html_parts.append("    <h2>Details for " + _html_escape(model) + "</h2>")
        if not rows:
            html_parts.append("    <div class='small'>No evaluation records for this model.</div>")
        else:
            html_parts.append("    <table>")
            html_parts.append("      <thead><tr><th>Abstract</th><th>Attribute</th><th>Result</th><th>Extracted Value</th><th>Expected/Matched Value</th><th>Match Method</th><th>Similarity</th><th>Vector Model Used</th><th>Vector Details</th></tr></thead>")
            html_parts.append("      <tbody>")
            for row in rows:
                tag_class = "hit" if row["result"] == "Hit" else ("miss" if row["result"] == "Miss" else "fp")
                similarity_display = "" if row.get("similarity") is None else f"{row['similarity']:.3f}"
                html_parts.append(
                    "        <tr>"
                    + "<td>" + _html_escape(row["abstract"]) + "</td>"
                    + "<td>" + _html_escape(row["attribute"]) + "</td>"
                    + "<td><span class='tag " + tag_class + "'>" + row["result"] + "</span></td>"
                    + "<td>" + _html_escape(row.get("extracted", "")) + "</td>"
                    + "<td>" + _html_escape(row.get("expected", "")) + "</td>"
                    + "<td>" + _html_escape(row.get("match_method", "")) + "</td>"
                    + "<td>" + similarity_display + "</td>"
                    + "<td>" + _html_escape(row.get("vector_model", "")) + "</td>"
                    + "<td>" + _html_escape(row.get("vector_details", "")) + "</td>"
                    + "</tr>"
                )
            html_parts.append("      </tbody>")
            html_parts.append("    </table>")
        html_parts.append("  </div>")

    html_parts.append("</body>")
    html_parts.append("</html>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    logger.info("HTML evaluation report saved to: %s", output_file)
    return output_file


def _html_escape(value: str) -> str:
    """Minimal HTML escaping for text content."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _collect_model_events(results: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Flatten hits/misses/false positives per model for HTML tables."""
    events: dict[str, list[dict[str, Any]]] = {}

    for abstract in results.get("abstracts", []):
        abstract_label = f"{abstract.get('id', 'unknown')} — {abstract.get('title', '')[:80]}"
        for model, model_result in abstract.get("models", {}).items():
            if "error" in model_result:
                continue

            evaluation = model_result.get("evaluation", {})
            events.setdefault(model, [])

            for hit in evaluation.get("hits", []):
                events[model].append({
                    "abstract": abstract_label,
                    "attribute": hit.get("attribute", ""),
                    "result": "Hit",
                    "extracted": hit.get("extracted_value", ""),
                    "expected": hit.get("matched_expected", ""),
                    "match_method": hit.get("match_method", ""),
                    "similarity": hit.get("similarity_score"),
                    "vector_model": hit.get("vector_selected_model", ""),
                    "vector_details": _format_vector_model_scores(hit.get("vector_model_scores")),
                })

            for miss in evaluation.get("misses", []):
                events[model].append({
                    "abstract": abstract_label,
                    "attribute": miss.get("attribute", ""),
                    "result": "Miss",
                    "extracted": "",
                    "expected": miss.get("expected_value", ""),
                    "match_method": "",
                    "similarity": None,
                    "vector_model": "",
                    "vector_details": "",
                })

            for fp in evaluation.get("false_positives", []):
                expected_vals = fp.get("expected_values", [])
                expected_str = ", ".join(expected_vals)
                events[model].append({
                    "abstract": abstract_label,
                    "attribute": fp.get("attribute", ""),
                    "result": "False Positive",
                    "extracted": fp.get("extracted_value", ""),
                    "expected": expected_str,
                    "match_method": "",
                    "similarity": fp.get("similarity_score"),
                    "vector_model": fp.get("vector_selected_model", ""),
                    "vector_details": _format_vector_model_scores(fp.get("vector_model_scores")),
                })

    for model in events:
        events[model].sort(key=lambda r: (r["abstract"], r["attribute"], r["result"]))

    return events


def _group_by_attribute(items: list[dict], key: str) -> dict[str, list[dict]]:
    """Group a list of dicts by a key value."""
    grouped: dict[str, list[dict]] = {}
    for item in items:
        attr = item.get(key, "unknown")
        if attr not in grouped:
            grouped[attr] = []
        grouped[attr].append(item)
    return grouped


def generate_abstract_evaluation_report(
    abstract_id: str,
    title: str,
    model: str,
    evaluation: dict[str, Any],
    output_dir: str,
    logger: logging.Logger = LOGGER,
) -> str:
    """Generate a per-abstract evaluation report.

    Args:
        abstract_id: Abstract identifier
        title: Abstract title
        model: Model name used for extraction
        evaluation: Evaluation results for this model on this abstract
        output_dir: Directory to save the report
        logger: Logger instance

    Returns:
        Path to the generated report file
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append(f"LLM EXTRACTION EVALUATION REPORT - {abstract_id}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Abstract: {title}")
    lines.append(f"Model: {model}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Evaluation configuration
    lines.append("-" * 60)
    lines.append("EVALUATION CONFIGURATION")
    lines.append("-" * 60)
    lines.append("")

    # Legend
    lines.append("-" * 60)
    lines.append("EVALUATION LEGEND:")
    lines.append("-" * 60)
    lines.append("")
    lines.append("  HIT           The LLM extracted a value that matches something in the gold standard")
    lines.append("  MISS          The gold standard expects a value, but the LLM didn't extract it")
    lines.append("  FALSE POSITIVE  The LLM extracted a value that is NOT in the gold standard")
    lines.append("")

    # Vector stats (if present)
    vector_stats = evaluation.get("vector_stats", {})
    if any(vector_stats.values()):
        lines.append("VECTOR EVALUATION STATS:")
        lines.append(f"  Auto-matches (score >= 0.85): {vector_stats.get('auto_matches', 0)}")
        lines.append(f"  Auto-rejects (score <= 0.50): {vector_stats.get('auto_rejects', 0)}")
        lines.append(f"  Ambiguous -> LLM: {vector_stats.get('ambiguous_to_llm', 0)}")
        if vector_stats.get("vector_unavailable", 0) > 0:
            lines.append(f"  Vector unavailable: {vector_stats.get('vector_unavailable', 0)}")
        lines.append("")

    # Overall scores
    scores = evaluation.get("scores", {})
    lines.append("-" * 60)
    lines.append("PERFORMANCE METRICS")
    lines.append("-" * 60)
    lines.append("")
    lines.append(f"  Total Expected Values: {scores.get('total_expected', 0)}")
    lines.append(f"  Total Hits:            {scores.get('total_hits', 0)}")
    lines.append(f"  Total Misses:          {scores.get('total_misses', 0)}")
    lines.append(f"  Total False Positives: {scores.get('total_false_positives', 0)}")
    lines.append("")
    lines.append(f"  Recall:    {scores.get('recall', 0):.2%}")
    lines.append(f"  Precision: {scores.get('precision', 0):.2%}")
    lines.append(f"  F1 Score:  {scores.get('f1', 0):.2%}")
    lines.append("")

    # Detailed hits
    hits = evaluation.get("hits", [])
    if hits:
        lines.append("-" * 60)
        lines.append("HITS (LLM extracted value matches gold standard):")
        lines.append("")
        for hit in hits:
            score_str = f" [sim={hit['similarity_score']:.3f}]" if "similarity_score" in hit else ""
            method_str = f" ({hit['match_method']})" if "match_method" in hit else ""
            vector_model_str = f" [vector_model={hit['vector_selected_model']}]" if hit.get("vector_selected_model") else ""
            vector_details = _format_vector_model_scores(hit.get("vector_model_scores"))
            vector_details_str = f" [vector_details={vector_details}]" if vector_details else ""
            lines.append(
                f"  {hit['attribute']}: LLM='{hit['extracted_value']}' -> matched gold='{hit['matched_expected']}'"
                f"{method_str}{score_str}{vector_model_str}{vector_details_str}"
            )

    # Misses
    misses = evaluation.get("misses", [])
    if misses:
        lines.append("-" * 60)
        lines.append("MISSES (gold standard value NOT extracted by LLM):")
        lines.append("")
        for miss in misses:
            lines.append(f"  x {miss['attribute']}: gold='{miss['expected_value']}' was not extracted")

    # False positives
    false_positives = evaluation.get("false_positives", [])
    if false_positives:
        lines.append("-" * 60)
        lines.append("FALSE POSITIVES (LLM extracted value NOT in gold standard):")
        lines.append("")
        for fp in false_positives:
            expected_vals = ", ".join(fp.get("expected_values", []))
            score_str = f" [best_sim={fp['similarity_score']:.3f}]" if "similarity_score" in fp else ""
            vector_model_str = f" [vector_model={fp['vector_selected_model']}]" if fp.get("vector_selected_model") else ""
            vector_details = _format_vector_model_scores(fp.get("vector_model_scores"))
            vector_details_str = f" [vector_details={vector_details}]" if vector_details else ""
            lines.append(
                f"  ? {fp['attribute']}: LLM='{fp['extracted_value']}' not in gold=[{expected_vals}]"
                f"{score_str}{vector_model_str}{vector_details_str}"
            )

    # Write to file
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "evaluation_report.txt")
    report_content = "\n".join(lines)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("Abstract evaluation report saved to: %s", report_file)
    return report_file


def _aggregate_model_scores(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Aggregate scores across all abstracts for each model."""
    model_scores: dict[str, dict[str, Any]] = {}

    for abstract_result in results.get("abstracts", []):
        for model, model_result in abstract_result.get("models", {}).items():
            if "error" in model_result:
                continue

            if model not in model_scores:
                model_scores[model] = {
                    "total_hits": 0,
                    "total_misses": 0,
                    "total_false_positives": 0,
                    "total_expected": 0,
                    "abstracts_evaluated": 0,
                }

            scores = model_result.get("evaluation", {}).get("scores", {})
            model_scores[model]["total_hits"] += scores.get("total_hits", 0)
            model_scores[model]["total_misses"] += scores.get("total_misses", 0)
            model_scores[model]["total_false_positives"] += scores.get("total_false_positives", 0)
            model_scores[model]["total_expected"] += scores.get("total_expected", 0)
            model_scores[model]["abstracts_evaluated"] += 1

    return model_scores
