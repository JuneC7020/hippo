"""Aggregate benchmark rows into summary.md / summary.csv.

Medians, not means: a single run that wandered for eight steps should not move the headline.
The headline is the reduction in median input tokens per run versus mode `all` at the same
catalog size, next to the success rate, so a cheaper mode that stops solving tasks is visible.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

MODE_LABEL = {"all": "all (baseline)", "search": "search", "search_schema": "search_schema"}


def load_rows(path: Path | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "_meta" in obj:
            meta = obj["_meta"]
        else:
            rows.append(obj)
    return rows, meta


def _med(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _mean(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _rng(values: Iterable[Any]) -> tuple[float, float] | None:
    vals = [float(v) for v in values if v is not None]
    return (min(vals), max(vals)) if vals else None


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per (size, mode), sizes ascending, modes in benchmark order."""
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(int(r["size"]), str(r["mode"]))].append(r)
    order = {"all": 0, "search": 1, "search_schema": 2}
    out: list[dict[str, Any]] = []
    for (size, mode), rs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))
    ):
        ok = [r for r in rs if not r.get("error")]
        n = len(rs)
        rec: dict[str, Any] = {
            "size": size,
            "mode": mode,
            "n": n,
            "errors": n - len(ok),
            "success_rate": _mean(r["success"] for r in ok) if ok else 0.0,
            "tool_called_rate": _mean(r["expected_tool_called"] for r in ok) if ok else 0.0,
            "recall_at_k": _mean(r.get("recall_at_k") for r in ok),
            "input_tokens_med": _med(r["input_tokens"] for r in ok),
            "input_tokens_range": _rng(r["input_tokens"] for r in ok),
            "output_tokens_med": _med(r["output_tokens"] for r in ok),
            "schema_tokens_med": _med(r["tool_schema_tokens"] for r in ok),
            "result_tokens_med": _med(r["tool_result_tokens"] for r in ok),
            "search_result_tokens_med": _med(r.get("search_result_tokens") for r in ok),
            "cache_read_share": _share(ok),
            "steps_med": _med(r["steps"] for r in ok),
            "n_unknown_total": sum(int(r.get("n_unknown") or 0) for r in ok),
            "wall_ms_med": _med(r["wall_ms"] for r in ok),
            "llm_ms_med": _med(r.get("llm_ms") for r in ok),
            "retrieval_ms_med": _med(r.get("retrieval_ms") for r in ok),
            "cost_usd_total": _sum_or_none(r.get("cost_usd") for r in ok),
            "cost_usd_med": _med(r.get("cost_usd") for r in ok),
            "catalog_schema_tokens": _med(r.get("catalog_schema_tokens") for r in ok),
            "estimated": any(r.get("estimated") for r in ok),
        }
        out.append(rec)
    return out


def _share(rows: Sequence[dict[str, Any]]) -> float | None:
    inp = sum(int(r.get("input_tokens") or 0) for r in rows)
    cached = sum(int(r.get("cache_read") or 0) for r in rows)
    return cached / inp if inp else None


def _sum_or_none(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) if vals else None


def headline(agg: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per size: each non-baseline mode's reduction vs `all` in median input tokens and cost."""
    by_size: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for rec in agg:
        by_size[rec["size"]][rec["mode"]] = rec
    out: list[dict[str, Any]] = []
    for size in sorted(by_size):
        base = by_size[size].get("all")
        for mode, rec in by_size[size].items():
            if mode == "all" or base is None:
                continue
            out.append(
                {
                    "size": size,
                    "mode": mode,
                    "input_reduction": _reduction(
                        base["input_tokens_med"], rec["input_tokens_med"]
                    ),
                    "schema_reduction": _reduction(
                        base["schema_tokens_med"], rec["schema_tokens_med"]
                    ),
                    "cost_reduction": _reduction(base["cost_usd_med"], rec["cost_usd_med"]),
                    "success_delta": (rec["success_rate"] or 0) - (base["success_rate"] or 0),
                    "steps_delta": (rec["steps_med"] or 0) - (base["steps_med"] or 0),
                    "wall_ratio": (
                        rec["wall_ms_med"] / base["wall_ms_med"]
                        if rec["wall_ms_med"] and base["wall_ms_med"]
                        else None
                    ),
                }
            )
    return out


def _reduction(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return 1 - after / before


# --------------------------------------------------------------------------- rendering


def _pct(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:+.0f}%" if signed else f"{x * 100:.0f}%"


def _less(reduction: float | None) -> str:
    if reduction is None:
        return "n/a"
    return f"{reduction * 100:.0f}% less" if reduction >= 0 else f"{-reduction * 100:.0f}% more"


def _num(x: float | None, digits: int = 0) -> str:
    if x is None:
        return "n/a"
    return f"{x:,.{digits}f}"


def _usd(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"${x:.4f}" if x < 0.1 else f"${x:.3f}"


def render_markdown(
    rows: Sequence[dict[str, Any]], agg: Sequence[dict[str, Any]], meta: dict[str, Any]
) -> str:
    lines: list[str] = []
    model = meta.get("model", "?")
    lines.append(f"# Tool exposure benchmark: {model}")
    lines.append("")
    lines.append(
        f"{len(rows)} runs = {meta.get('n_tasks', '?')} tasks x modes {meta.get('modes')} "
        f"x catalog sizes {meta.get('sizes')} x {meta.get('repeats', 1)} repeat(s). "
        f"Retriever `{meta.get('retriever', '?')}`, k={meta.get('search_k', '?')}, "
        f"max_steps={meta.get('max_steps', '?')}, always_on={meta.get('always_on') or '[]'}."
    )
    if meta.get("fake_llm"):
        lines.append("")
        lines.append(
            "**Scripted fake LLM; token figures are tiktoken estimates, not billed usage.**"
        )
    lines.append("")
    lines.append("Medians per run unless marked. `schema` = tool-definition tokens summed over the")
    lines.append("run's requests (what `all` re-sends every step); `results` = tool result tokens")
    lines.append("entering the history; `cached` = share of input tokens the provider served from")
    lines.append("its prompt cache. Success = expected tool called with the expected arguments.")
    lines.append("")

    head = headline(agg)
    if head:
        lines.append("## Headline: before/after vs `all` at the same catalog size")
        lines.append("")
        lines.append(
            "| tools | mode | input tokens | schema tokens | cost | success | steps | wall |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for h in head:
            lines.append(
                f"| {h['size']} | {h['mode']} | {_less(h['input_reduction'])} | "
                f"{_less(h['schema_reduction'])} | {_less(h['cost_reduction'])} | "
                f"{_pct(h['success_delta'], signed=True)} | {h['steps_delta']:+.1f} | "
                f"{'x' + format(h['wall_ratio'], '.2f') if h['wall_ratio'] else 'n/a'} |"
            )
        lines.append("")

    lines.append("## Per cell")
    lines.append("")
    lines.append(
        "| tools | mode | n | success | tool called | recall@k | input (med) | input (range) | "
        "output | schema | results | cached | steps | unknown | wall ms | cost/run | cost total |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in agg:
        rng = a["input_tokens_range"]
        lines.append(
            f"| {a['size']} | {MODE_LABEL.get(a['mode'], a['mode'])} | {a['n']}"
            + (f" ({a['errors']} err)" if a["errors"] else "")
            + f" | {_pct(a['success_rate'])} | {_pct(a['tool_called_rate'])} | "
            f"{_pct(a['recall_at_k'])} | {_num(a['input_tokens_med'])} | "
            f"{_num(rng[0]) + '-' + _num(rng[1]) if rng else 'n/a'} | "
            f"{_num(a['output_tokens_med'])} | {_num(a['schema_tokens_med'])} | "
            f"{_num(a['result_tokens_med'])} | {_pct(a['cache_read_share'])} | "
            f"{_num(a['steps_med'], 1)} | {a['n_unknown_total']} | {_num(a['wall_ms_med'])} | "
            f"{_usd(a['cost_usd_med'])} | {_usd(a['cost_usd_total'])} |"
        )
    lines.append("")

    sizes = sorted({a["size"] for a in agg})
    if sizes:
        lines.append("## Catalog schema block per request (mode `all` binds all of it)")
        lines.append("")
        lines.append("| tools | schema tokens per request |")
        lines.append("|---|---|")
        for s in sizes:
            rec = next((a for a in agg if a["size"] == s and a["catalog_schema_tokens"]), None)
            lines.append(f"| {s} | {_num(rec['catalog_schema_tokens']) if rec else 'n/a'} |")
        lines.append("")

    failures = [r for r in rows if not r.get("success")]
    if failures:
        lines.append("## Runs that did not score")
        lines.append("")
        lines.append("| task | mode | tools | why |")
        lines.append("|---|---|---|---|")
        for r in failures[:60]:
            if r.get("error"):
                why = "error: " + str(r["error"])[:120]
            elif not r.get("expected_tool_called"):
                why = "expected tool never called"
                if r.get("recall_at_k") is not None and r["recall_at_k"] < 1:
                    why += f" (recall@k={r['recall_at_k']:.2f})"
                if r.get("n_unknown"):
                    why += f", {r['n_unknown']} unknown-tool call(s)"
            else:
                why = "called with unexpected args: " + str(r.get("matched_call"))[:120]
            lines.append(f"| {r['task_id']} | {r['mode']} | {r['size']} | {why} |")
        if len(failures) > 60:
            lines.append(f"| ... | | | {len(failures) - 60} more |")
        lines.append("")
    return "\n".join(lines) + "\n"


CSV_FIELDS = [
    "size",
    "mode",
    "n",
    "errors",
    "success_rate",
    "tool_called_rate",
    "recall_at_k",
    "input_tokens_med",
    "output_tokens_med",
    "schema_tokens_med",
    "result_tokens_med",
    "search_result_tokens_med",
    "cache_read_share",
    "steps_med",
    "n_unknown_total",
    "wall_ms_med",
    "llm_ms_med",
    "retrieval_ms_med",
    "cost_usd_med",
    "cost_usd_total",
    "catalog_schema_tokens",
    "estimated",
]


def write_csv(agg: Sequence[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for a in agg:
            w.writerow({k: a.get(k) for k in CSV_FIELDS})
    return path


def build_report(runs_path: Path, out_dir: Path | None = None) -> tuple[Path, Path]:
    rows, meta = load_rows(runs_path)
    agg = aggregate(rows)
    out_dir = out_dir or runs_path.parent
    md = out_dir / "summary.md"
    md.write_text(render_markdown(rows, agg, meta), encoding="utf-8")
    csv_path = write_csv(agg, out_dir / "summary.csv")
    return md, csv_path
