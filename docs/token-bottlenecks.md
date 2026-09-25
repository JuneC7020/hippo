# Token bottlenecks with many tools

Written *before* running the benchmark, so the measured numbers can confirm or refute it. Each entry
says what the bottleneck is, whether it is a fact in the current code or a hypothesis, how large we
expect it to be, and which metric field will settle it.

Target setup: DeepSeek v4 flash 0731 over OpenRouter, single-agent `tool_loop`, catalog of ~180
tools across ~15 servers. Estimates below assume an average tool schema of ~160 tokens, so the full
tool-definitions block is ~29k tokens.

## Cost model

Per request, the input is:

```
system prompt + recall block + task + message history (incl. accumulated tool results)
  + tool_definitions block
```

The chat API is stateless, so **everything in that sum is re-sent on every step**. Total run cost is
therefore roughly `steps x (fixed prefix + growing history + tool_definitions)`. Almost every
bottleneck below is a multiplication by `steps` that we failed to notice.

Baseline for a task that needs one real tool call plus an answer (2 requests):

| mode | requests | schemas per request | total input |
|---|---|---|---|
| `all` | 2 | ~29k | ~59k |
| `search` (search -> load -> call -> answer) | 4 | ~0.25k, then ~0.4k | ~4k |
| `search_schema` (with the duplication in B4) | 3 | ~0.25k + 0.8k of result text | ~5.4k |

## B1. Tool-definitions block multiplied by steps

**Fact.** `tool_loop` binds once before the loop and never changes the set
([`../src/hippo/agent/graph.py`](../src/hippo/agent/graph.py), line 217):

```python
bound = llm.bind_tools(tools) if tools else llm
by_name = {t.name: t for t in tools}
```

The dominant bottleneck, and the only reason the project exists. At 180 tools a 6-step worker pays
~29k tokens six times over for definitions it uses two of. Cost is `O(catalog x steps)` and the
useful fraction is `O(1)`.

Expected size: 85-95 percent of input tokens at 180 tools; ~50 percent at 25 tools.

Measured by: `tool_schema_tokens`, `n_bound_tools` per step.

## B2. Prompt-cache invalidation by dynamic binding

**Hypothesis, and the one most likely to undercut the headline number.** In the baseline the 29k
definitions block is a stable prefix across all steps of all tasks, which is the ideal shape for
provider-side prompt caching (DeepSeek charges cache reads at roughly a tenth of cache misses). Tool
search changes the bound set between steps, so the prefix changes and the cache is invalidated
every time a tool is activated.

The consequence: a 15x reduction in *tokens* may be only a 4-8x reduction in *dollars*, and the
baseline's effective cost is much lower than its raw token count suggests. Reporting tokens without
cache hits would overstate the win.

Expected size: could reclaim 60-80 percent of the baseline's apparent cost.

Measured by: `cache_read` from `usage_metadata`, reported alongside `input_tokens`, plus a
`cost_usd` column computed from a price table with separate hit/miss rates.

Mitigations to evaluate:

- Batch activations so the tool set changes once per run rather than once per step.
- Never deactivate within a run (trades against B5).
- The generic-dispatcher design in B4, which keeps the definitions block byte-identical forever.

## B3. Tool-result accumulation

**Fact, and the next bottleneck once B1 is removed.** Results are capped at 12k characters in two
places, roughly 3k tokens per call, and every result stays in the history to be re-sent:

```python
result = str(result)[:12_000]                    # agent/graph.py:264, in tool_loop
text = _result_to_text(result)[:12_000]          # tools/mcp_client.py:172, in McpHub._wrap
```

With schemas gone, four tool calls of 3k tokens each dominate a run. This is why the benchmark must
split `tool_schema_tokens` from `tool_result_tokens` instead of reporting one `input_tokens` figure;
otherwise the second round of optimization has nothing to aim at.

Expected size: 40-70 percent of input tokens in `search` mode.

Measured by: `tool_result_tokens`, and result character counts already in the trace.

## B4. Round-trip overhead, and schema duplication in `search_schema`

**Fact of the design.** Mode `search` adds two round trips (`tool_search`, then `tool_load`) before
any real work. Each extra round trip re-sends the whole prompt and produces output tokens, so the
saving is not free and wall-clock time roughly doubles.

Two specific traps:

1. **Break-even.** Below roughly 10 tools, `search` costs more tokens than `all`; on latency the
   crossover is much higher because 2 requests become 4. The 25/75/180 sweep exists to find this
   point, and the honest conclusion may be an adaptive policy: use `all` while
   `schema_tokens(catalog)` is under a threshold, else switch to search.
2. **Duplication in `search_schema`.** If search returns full schemas as text *and* auto-activates
   them, the same schema is paid for twice: once in the `ToolMessage` (which then persists in
   history for every later step) and once in the definitions block. The cheap version of this mode
   must return names only and rely on activation. That makes `search_schema` simply `search` minus
   the `tool_load` step, which is the honest description of it.

A third design worth benchmarking: a generic `invoke_tool(name, args)` dispatcher. The definitions
block then holds only 2-3 meta-tools and never changes, so the cache prefix is perfect (fixes B2)
and discovered schemas land at the tail of the history where they extend the cache instead of
invalidating it. The cost is losing native schema validation, so argument error rates should rise;
that trade is measurable.

Measured by: `steps`, `n_search_calls`, `wall_ms`, `expected_tool_called`.

## B5. Activated-tool creep

**Hypothesis.** Nothing in the plan removes a tool from the activated set. A tool activated at step 1
is re-sent at every step to the end of the run, so a long run drifts back toward baseline cost. A
worker that searches three times ends up carrying ~15 schemas.

Mitigations: cap the activated set (LRU, e.g. 5), or drop a tool after one successful use. Both
fight B2, since every eviction is another cache invalidation. The right answer is probably a cap
high enough to be hit rarely.

Measured by: `n_bound_tools` per step, plotted against step index.

## B6. Compression budget blind to schemas

**Fact, and a live bug independent of tool search.** `count_tokens()` sums messages only; it never
sees the tool-definitions block. With 180 tools, `HIPPO_TOKEN_BUDGET=8000` is judging about a
quarter of the real prompt, so compression fires far too late (or never) exactly when it matters
most. Passing a `fixed_overhead` into `count_tokens` / `compress_messages` fixes the baseline too,
which means the before/after comparison should be run against the *fixed* baseline to stay honest.

Measured by: `compress` trace events per run, and `before_tokens` vs the true prompt size.

## B7. Search-result text in history

**Hypothesis, small but avoidable.** Each `tool_search` result is a `ToolMessage` that stays in
history and is re-sent every later step. Ten candidates with two-line descriptions is ~400 tokens
re-sent five times. Keep summaries to one line under roughly 15 words, and let compression fold old
search results away first, since they are the most disposable thing in the history.

Note the interaction with `KEEP_TAIL = 4` in
[`../src/hippo/agent/context.py`](../src/hippo/agent/context.py), the number of recent messages that
are never compressed. A search-load-call sequence is four messages, so a compression triggered
mid-sequence can protect exactly the search noise we would most like to drop.

Measured by: `search_result_tokens`, if worth splitting out from `tool_result_tokens`.

## B8. Retrieval misses turn into wasted steps

**Hypothesis.** A miss is more expensive than it looks: the model searches again with a reworded
query, or worse, hallucinates a tool name and burns a step on `unknown tool: ...`. Two wasted steps
at ~1k tokens each is cheap in `search` mode but the step budget is what actually runs out, so a
recall failure can turn into a task failure rather than a cost increase.

This is why the scoring is `recall@k` separately from `expected_tool_called`: it distinguishes "the
retriever never surfaced it" from "the retriever surfaced it and the model picked wrong."

Expected: keyword retrieval to beat embeddings on this catalog, because tool names share vocabulary
with the queries. Worth being wrong about.

Measured by: `recall_at_k`, `expected_tool_called`, unknown-tool counts in the trace.

## B9. Retriever-side cost

**Hypothesis, low impact but easy to forget.** `EmbeddingRetriever` needs one embedding call per
search, adding latency and a second billed model to the run, plus a one-time index build over ~180
descriptions. `KeywordRetriever` is free and deterministic. If the two are close on `recall@k`, the
keyword version is strictly better for a benchmark that has to be reproducible.

Measured by: `wall_ms` split into LLM time and retrieval time.

## B10. Nested schema verbosity

**Hypothesis.** `json_schema_to_model` in [../src/hippo/tools/mcp_client.py](../src/hippo/tools/mcp_client.py)
deliberately expands nested objects and arrays into real pydantic models, because flattening them
made the model invent argument keys. Correct, but it means one tool with a nested `edits:
[{oldText, newText}]` argument can cost several hundred tokens on its own. Per-tool schema size will
have a long tail, so the "average 160 tokens" assumption needs checking, and a handful of fat tools
may dominate the activated-set cost.

A slimming pass (drop descriptions on optional parameters, or omit optional parameters entirely on
first activation) is measurable but risks argument errors.

Measured by: per-tool `schema_tokens` distribution, p50 vs p95.

## Priority

| | bottleneck | expected impact | confidence |
|---|---|---|---|
| 1 | B1 definitions x steps | very high | certain |
| 2 | B2 cache invalidation | high, works against us | medium |
| 3 | B3 result accumulation | high, after B1 | certain |
| 4 | B4 round trips and duplication | medium | certain |
| 5 | B6 compression blind to schemas | medium | certain |
| 6 | B5 activated-tool creep | medium | medium |
| 7 | B8 retrieval misses | medium, as failures not cost | low |
| 8 | B10 schema verbosity | low to medium | low |
| 9 | B7 search text in history | low | medium |
| 10 | B9 retriever cost | low | medium |

## Metrics this implies

Beyond the fields already in the plan, the analysis above says the runner must record
`tool_schema_tokens` and `tool_result_tokens` separately (B3), `cache_read` and `cost_usd` with
split hit/miss pricing (B2), `n_bound_tools` per step rather than per run (B5), and retrieval time
separately from LLM time (B9). Without those splits the benchmark can show that tool search is
cheaper but not why, or what to fix next.

One tuning knob deserves calling out because the plan currently defaults it to empty for
measurement cleanliness: `always_on`. The existing `BASELINE_TOOLS` set in
[../src/hippo/agent/graph.py](../src/hippo/agent/graph.py) already encodes which tools are used on
almost every task. Putting those in `always_on` costs about 1k tokens of permanent schema and saves
two round trips on the majority of tasks, so the realistic configuration is very likely
`always_on = BASELINE_TOOLS` with search covering the long tail.
