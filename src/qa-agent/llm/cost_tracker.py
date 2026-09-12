"""
Tracks token usage and estimated cost per LLM call, so a run's total LLM
spend is visible and boundable — `provider_router.py` records into this
after every completion, and `api/routers/runs.py` can surface a run's
total cost alongside its result.

Pricing is necessarily approximate and out of date the moment a provider
changes it; treat `estimate_cost_usd` as a budgeting signal, not a bill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from qa_agent.config.logging_config import get_logger

logger = get_logger(__name__)

# USD per 1M tokens, (input, output). Approximate; update as providers
# change pricing. An unrecognized model falls back to a conservative
# default rather than silently reporting $0.
_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "claude-opus-5": (15.00, 75.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}
_DEFAULT_PRICING = (5.00, 15.00)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = _PRICING_PER_MILLION_TOKENS.get(model, _DEFAULT_PRICING)
    return round((input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate, 6)


@dataclass(frozen=True)
class LLMCallRecord:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    run_id: str | None
    purpose: str  # e.g. "story_parsing", "test_planning", "root_cause_analysis"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CostSummary:
    call_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


class BudgetExceededError(Exception):
    def __init__(self, run_id: str, spent_usd: float, budget_usd: float) -> None:
        self.run_id = run_id
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd
        super().__init__(
            f"Run {run_id} has spent ${spent_usd:.4f}, exceeding its budget of ${budget_usd:.4f}."
        )


class CostTracker:
    """
    Example:
        tracker = CostTracker(budget_usd_per_run=0.50)
        tracker.record(
            provider="anthropic", model="claude-sonnet-5",
            input_tokens=1200, output_tokens=300,
            run_id="run-123", purpose="test_planning",
        )
        print(tracker.summary(run_id="run-123"))
    """

    def __init__(self, budget_usd_per_run: float | None = None) -> None:
        self._budget_usd_per_run = budget_usd_per_run
        self._records: list[LLMCallRecord] = []

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str,
        run_id: str | None = None,
        cost_usd: float | None = None,
    ) -> LLMCallRecord:
        """
        `cost_usd`, if the provider's API returned an authoritative
        figure, is used as-is; otherwise cost is estimated from the
        static pricing table above.
        """
        resolved_cost = cost_usd if cost_usd is not None else estimate_cost_usd(
            model, input_tokens, output_tokens
        )
        record = LLMCallRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=resolved_cost,
            run_id=run_id,
            purpose=purpose,
        )
        self._records.append(record)
        logger.debug(
            "LLM call recorded: provider=%s model=%s purpose=%s tokens=%d/%d cost=$%.6f",
            provider,
            model,
            purpose,
            input_tokens,
            output_tokens,
            resolved_cost,
        )

        if run_id is not None and self._budget_usd_per_run is not None:
            spent = self.summary(run_id=run_id).total_cost_usd
            if spent > self._budget_usd_per_run:
                raise BudgetExceededError(run_id, spent, self._budget_usd_per_run)

        return record

    def summary(self, run_id: str | None = None, purpose: str | None = None) -> CostSummary:
        matching = [
            r
            for r in self._records
            if (run_id is None or r.run_id == run_id) and (purpose is None or r.purpose == purpose)
        ]
        return CostSummary(
            call_count=len(matching),
            total_input_tokens=sum(r.input_tokens for r in matching),
            total_output_tokens=sum(r.output_tokens for r in matching),
            total_cost_usd=round(sum(r.cost_usd for r in matching), 6),
        )

    def records_for_run(self, run_id: str) -> list[LLMCallRecord]:
        return [r for r in self._records if r.run_id == run_id]

    def clear(self) -> None:
        self._records.clear()