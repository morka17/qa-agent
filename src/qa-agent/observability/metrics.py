"""
Prometheus counters/histograms for the agent pipeline. Import
`record_run_finished`, `record_step`, `record_bug_filed`, and
`record_llm_call` from wherever those events actually happen
(`orchestration/agent_loop.py`, `orchestration/task_queue.py`) rather
than instantiating metric objects ad hoc elsewhere — every metric this
module exposes is defined exactly once here, so `/metrics` output is
consistent regardless of which module triggered an update.

`api/app.py` mounts `metrics_asgi_app()` at `/metrics` for Prometheus to
scrape; `observability/dashboards/*.json` are pre-built Grafana panels
over exactly these metric names.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# --- Run-level metrics ---

RUNS_TOTAL = Counter(
    "sentinel_runs_total",
    "Total number of agent runs, by final state.",
    labelnames=["state"],
)

RUN_DURATION_SECONDS = Histogram(
    "sentinel_run_duration_seconds",
    "Wall-clock duration of a full agent run, from ingestion to its terminal state.",
    labelnames=["final_state"],
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800),
)

# --- Step-level metrics ---

STEPS_TOTAL = Counter(
    "sentinel_steps_total",
    "Total number of executed plan steps, by action type and outcome.",
    labelnames=["action", "success"],
)

STEP_DURATION_SECONDS = Histogram(
    "sentinel_step_duration_seconds",
    "Duration of a single step execution (resolve + act, including any self-healing retries).",
    labelnames=["action"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30),
)

ELEMENT_RESOLUTION_TOTAL = Counter(
    "sentinel_element_resolution_total",
    "Element resolutions, by strategy used (cache_hit/heuristic/visual_grounding).",
    labelnames=["strategy"],
)

# --- Bug / triage metrics ---

BUGS_FILED_TOTAL = Counter(
    "sentinel_bugs_filed_total",
    "Bugs filed against an issue tracker, by severity and category.",
    labelnames=["severity", "category", "tracker"],
)

FAILURE_CLASSIFICATIONS_TOTAL = Counter(
    "sentinel_failure_classifications_total",
    "Failure classifications produced by the triage classifier, by category.",
    labelnames=["category"],
)

# --- LLM metrics ---

LLM_CALLS_TOTAL = Counter(
    "sentinel_llm_calls_total",
    "LLM completion calls, by provider, model, and purpose.",
    labelnames=["provider", "model", "purpose"],
)

LLM_CALL_DURATION_SECONDS = Histogram(
    "sentinel_llm_call_duration_seconds",
    "Duration of a single LLM completion call.",
    labelnames=["provider", "model"],
    buckets=(0.5, 1, 2, 5, 10, 20, 45),
)

LLM_COST_USD_TOTAL = Counter(
    "sentinel_llm_cost_usd_total",
    "Estimated cumulative LLM spend in USD, by provider and model.",
    labelnames=["provider", "model"],
)


def record_run_finished(final_state: str, duration_seconds: float) -> None:
    RUNS_TOTAL.labels(state=final_state).inc()
    RUN_DURATION_SECONDS.labels(final_state=final_state).observe(duration_seconds)


def record_step(action: str, success: bool, duration_seconds: float) -> None:
    STEPS_TOTAL.labels(action=action, success=str(success)).inc()
    STEP_DURATION_SECONDS.labels(action=action).observe(duration_seconds)


def record_element_resolution(strategy: str) -> None:
    ELEMENT_RESOLUTION_TOTAL.labels(strategy=strategy).inc()


def record_bug_filed(severity: str, category: str, tracker: str) -> None:
    BUGS_FILED_TOTAL.labels(severity=severity, category=category, tracker=tracker).inc()


def record_failure_classification(category: str) -> None:
    FAILURE_CLASSIFICATIONS_TOTAL.labels(category=category).inc()


def record_llm_call(
    provider: str, model: str, purpose: str, duration_seconds: float, cost_usd: float
) -> None:
    LLM_CALLS_TOTAL.labels(provider=provider, model=model, purpose=purpose).inc()
    LLM_CALL_DURATION_SECONDS.labels(provider=provider, model=model).observe(duration_seconds)
    LLM_COST_USD_TOTAL.labels(provider=provider, model=model).inc(cost_usd)


def metrics_payload() -> tuple[bytes, str]:
    """Returns `(body, content_type)` ready to hand back from a `/metrics` route."""
    return generate_latest(), CONTENT_TYPE_LATEST