"""The benchmark grid: mode x catalog size x repeat x task, one JSONL row per run.

Each run is a fresh single-agent `run_tool_agent` over a synthetic catalog subset that is
identical for every mode at a given size (same seed, all expected tools included). Token
numbers come from the provider's usage block via `UsageMeter`; scoring from `bench.tasks`.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hippo.bench.catalog import CatalogSpec, ToolSpec, load_catalog, make_tools, select_subset
from hippo.bench.tasks import Task, load_tasks, required_tools, score_task
from hippo.metrics import Prices
from hippo.tools.registry import ToolRegistry
from hippo.tools.search import normalize_exposure
from hippo.trace import Tracer

DEFAULT_MODES = ("all", "search", "search_schema")
DEFAULT_SIZES = (25, 75, 183)


@dataclass
class BenchConfig:
    out_dir: Path
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str | None = None
    llm_factory: Callable[[], Any] | None = None  # tests inject a fake; None = real model
    modes: Sequence[str] = DEFAULT_MODES
    sizes: Sequence[int] = DEFAULT_SIZES
    repeats: int = 1
    tasks: Sequence[Task] = ()
    catalog: CatalogSpec | None = None
    retriever: str = "keyword"
    search_k: int = 5
    always_on: Sequence[str] = ()
    max_steps: int = 8
    max_active: int | None = None
    concurrency: int = 2
    seed: int = 7
    prices: Prices | None = None
    tool_budget: int = 40
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.modes = [normalize_exposure(m) for m in self.modes]
        self.sizes = [int(s) for s in self.sizes]
        if self.catalog is None:
            self.catalog = load_catalog()
        if not self.tasks:
            self.tasks = load_tasks()
        self.out_dir = Path(self.out_dir)

    def meta(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "modes": list(self.modes),
            "sizes": list(self.sizes),
            "repeats": self.repeats,
            "n_tasks": len(self.tasks),
            "catalog": str(self.catalog.path) if self.catalog and self.catalog.path else None,
            "catalog_tools": len(self.catalog) if self.catalog else 0,
            "retriever": self.retriever,
            "search_k": self.search_k,
            "always_on": list(self.always_on),
            "max_steps": self.max_steps,
            "max_active": self.max_active,
            "seed": self.seed,
            "fake_llm": self.llm_factory is not None,
            **self.extra,
        }


@dataclass
class RunRow:
    task_id: str
    mode: str
    size: int
    repeat: int
    model: str
    retriever: str
    search_k: int
    # scoring
    success: bool = False
    expected_tool_called: bool = False
    args_ok: bool = False
    recall_at_k: float | None = None
    # shape of the run
    steps: int = 0
    n_tool_calls: int = 0
    n_search_calls: int = 0
    n_load_calls: int = 0
    n_unknown: int = 0
    reason: str = ""
    # tokens
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    tool_schema_tokens: int = 0
    tool_result_tokens: int = 0
    search_result_tokens: int = 0
    n_bound_tools_per_step: list[int] = field(default_factory=list)
    catalog_schema_tokens: int = 0  # what mode `all` binds per step at this size
    estimated: bool = False
    # time and money
    wall_ms: float = 0.0
    llm_ms: float = 0.0
    retrieval_ms: float = 0.0
    cost_usd: float | None = None
    # debugging
    matched_call: str | None = None
    answer: str = ""
    error: str | None = None
    trace: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def catalog_subset(cfg: BenchConfig, size: int) -> list[ToolSpec]:
    assert cfg.catalog is not None
    return select_subset(cfg.catalog, size, must_include=required_tools(cfg.tasks), seed=cfg.seed)


async def run_one(
    cfg: BenchConfig,
    task: Task,
    mode: str,
    size: int,
    repeat: int,
    specs: Sequence[ToolSpec],
    *,
    catalog_schema_tokens: int = 0,
) -> RunRow:
    from hippo.agent.graph import run_tool_agent

    row = RunRow(
        task_id=task.id,
        mode=mode,
        size=size,
        repeat=repeat,
        model=cfg.model,
        retriever=cfg.retriever,
        search_k=cfg.search_k,
        catalog_schema_tokens=catalog_schema_tokens,
    )
    trace_path = cfg.out_dir / "traces" / f"{task.id}__{mode}__{size}__r{repeat}.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    tracer = Tracer(trace_path)
    row.trace = str(trace_path)
    registry = ToolRegistry(max_calls=cfg.tool_budget, allow_dangerous=True)
    calls: list[dict[str, Any]] = []
    tools = make_tools(specs, registry=registry, tracer=tracer, calls=calls)
    t0 = time.perf_counter()
    try:
        run = await run_tool_agent(
            task.prompt,
            tools=tools,
            tracer=tracer,
            llm=cfg.llm_factory() if cfg.llm_factory else None,
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            max_steps=cfg.max_steps,
            exposure=mode,
            retriever=cfg.retriever,
            search_k=cfg.search_k,
            always_on=cfg.always_on,
            registry=registry,
            index_dir=cfg.out_dir / "tool_index" if cfg.retriever == "embedding" else None,
            max_active=cfg.max_active,
            label=f"{task.id}/{mode}/{size}",
        )
    except Exception as exc:  # noqa: BLE001 - one failed run must not kill the grid
        row.wall_ms = (time.perf_counter() - t0) * 1000
        row.error = f"{type(exc).__name__}: {exc}"[:500]
        tracer.emit("run_error", error=row.error, tb=traceback.format_exc()[-2000:])
        return row
    row.wall_ms = (time.perf_counter() - t0) * 1000

    events = tracer.read()
    searches = [e for e in events if e.get("kind") == "tool_search"]
    loads = [e for e in events if e.get("kind") == "tool_load"]
    unknown = [e for e in events if e.get("kind") == "unknown_tool"]
    score = score_task(task, calls, searches, n_load_calls=len(loads), n_unknown=len(unknown))
    usage = run.meter.as_dict(cfg.prices)

    row.success = score.success
    row.expected_tool_called = score.expected_tool_called
    row.args_ok = score.args_ok
    row.recall_at_k = score.recall_at_k
    row.n_tool_calls = score.n_tool_calls
    row.n_search_calls = score.n_search_calls
    row.n_load_calls = score.n_load_calls
    row.n_unknown = score.n_unknown
    row.matched_call = score.as_dict()["matched_call"]
    row.reason = run.reason
    row.steps = usage["n_requests"]
    row.input_tokens = usage["input_tokens"]
    row.output_tokens = usage["output_tokens"]
    row.cache_read = usage["cache_read"]
    row.tool_schema_tokens = usage["tool_schema_tokens"]
    row.tool_result_tokens = usage["tool_result_tokens"]
    row.search_result_tokens = usage["search_result_tokens"]
    row.n_bound_tools_per_step = usage["n_bound_tools_per_step"]
    row.estimated = usage["estimated"]
    row.llm_ms = usage["llm_ms"]
    row.cost_usd = usage["cost_usd"]
    if run.exposure.catalog is not None:
        row.retrieval_ms = round(run.exposure.catalog.search_ms, 2)
    row.answer = run.text[:300]
    return row


async def run_grid(
    cfg: BenchConfig,
    *,
    progress: Callable[[RunRow, int, int], None] | None = None,
    limit: int | None = None,
) -> list[RunRow]:
    """Run every (mode, size, repeat, task) and return rows in a stable order."""
    from hippo.metrics import schema_tokens

    tasks = list(cfg.tasks)[: limit or None]
    subsets = {size: catalog_subset(cfg, size) for size in cfg.sizes}
    catalog_tokens = {
        size: schema_tokens(make_tools(specs), cfg.model) for size, specs in subsets.items()
    }
    jobs: list[tuple[str, int, int, Task]] = [
        (mode, size, rep, task)
        for size in cfg.sizes
        for mode in cfg.modes
        for rep in range(cfg.repeats)
        for task in tasks
    ]
    sem = asyncio.Semaphore(max(1, cfg.concurrency))
    done = 0
    rows: list[RunRow | None] = [None] * len(jobs)

    async def _job(i: int, mode: str, size: int, rep: int, task: Task) -> None:
        nonlocal done
        async with sem:
            row = await run_one(
                cfg,
                task,
                mode,
                size,
                rep,
                subsets[size],
                catalog_schema_tokens=catalog_tokens[size],
            )
        rows[i] = row
        done += 1
        if progress:
            progress(row, done, len(jobs))

    await asyncio.gather(*(_job(i, *job) for i, job in enumerate(jobs)))
    return [r for r in rows if r is not None]


def write_rows(rows: Iterable[RunRow], path: Path, meta: dict[str, Any] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if meta is not None:
            f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row.as_dict(), ensure_ascii=False, default=str) + "\n")
    return path
