"""
Defines the valid states a single test run moves through and enforces
legal transitions between them, so `agent_loop.py` can't accidentally
(e.g. after a bug in error-handling) leave a run in an inconsistent state
like "EXECUTING" forever, or jump straight from "PLANNING" to "PASSED"
without ever executing anything.

Kept deliberately dependency-free (no Playwright/LLM/DB imports) so it's
trivially unit-testable and reusable anywhere a run's lifecycle needs to
be tracked or displayed (e.g. the FastAPI control-plane API's run status
endpoint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class RunState(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    TRIAGING = "triaging"
    REPORTING = "reporting"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    CANCELLED = "cancelled"


# Terminal states: once reached, no further transition is valid (a new
# run must be started instead of resuming this one).
_TERMINAL_STATES = frozenset(
    {RunState.PASSED, RunState.FAILED, RunState.ERROR, RunState.CANCELLED}
)

# The full legal transition graph. Anything not listed here as a target
# of the current state is rejected — this is what turns "the loop skipped
# a phase" from a silent bug into an immediate, loud exception.
_ALLOWED_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.PENDING: frozenset({RunState.INGESTING, RunState.CANCELLED, RunState.ERROR}),
    RunState.INGESTING: frozenset({RunState.PLANNING, RunState.CANCELLED, RunState.ERROR}),
    RunState.PLANNING: frozenset(
        {RunState.AWAITING_APPROVAL, RunState.EXECUTING, RunState.CANCELLED, RunState.ERROR}
    ),
    RunState.AWAITING_APPROVAL: frozenset(
        {RunState.EXECUTING, RunState.CANCELLED, RunState.ERROR}
    ),
    RunState.EXECUTING: frozenset({RunState.VERIFYING, RunState.CANCELLED, RunState.ERROR}),
    RunState.VERIFYING: frozenset(
        {RunState.PASSED, RunState.TRIAGING, RunState.CANCELLED, RunState.ERROR}
    ),
    RunState.TRIAGING: frozenset({RunState.REPORTING, RunState.CANCELLED, RunState.ERROR}),
    RunState.REPORTING: frozenset({RunState.FAILED, RunState.CANCELLED, RunState.ERROR}),
    # Terminal states have no outgoing transitions.
    RunState.PASSED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.ERROR: frozenset(),
    RunState.CANCELLED: frozenset(),
}


class InvalidStateTransitionError(Exception):
    def __init__(self, current: RunState, attempted: RunState) -> None:
        self.current = current
        self.attempted = attempted
        allowed = sorted(s.value for s in _ALLOWED_TRANSITIONS.get(current, frozenset()))
        super().__init__(
            f"Cannot transition from {current.value!r} to {attempted.value!r}. "
            f"Allowed transitions from {current.value!r}: {allowed or '(none — terminal state)'}."
        )


@dataclass(frozen=True)
class StateTransition:
    from_state: RunState
    to_state: RunState
    timestamp: datetime
    reason: str | None = None


@dataclass
class RunStateMachine:
    """
    Example:
        machine = RunStateMachine(run_id="run-123")
        machine.transition(RunState.INGESTING)
        machine.transition(RunState.PLANNING)
        ...
        machine.transition(RunState.PASSED)
        print(machine.history)
    """

    run_id: str
    state: RunState = RunState.PENDING
    history: list[StateTransition] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def can_transition_to(self, target: RunState) -> bool:
        return target in _ALLOWED_TRANSITIONS.get(self.state, frozenset())

    def transition(self, target: RunState, reason: str | None = None) -> None:
        if not self.can_transition_to(target):
            raise InvalidStateTransitionError(self.state, target)

        self.history.append(
            StateTransition(
                from_state=self.state,
                to_state=target,
                timestamp=datetime.now(timezone.utc),
                reason=reason,
            )
        )
        self.state = target

    def duration_in_state_ms(self) -> float | None:
        """Milliseconds spent in the current state so far, or None if no transition has happened yet."""
        if not self.history:
            return None
        last = self.history[-1]
        return (datetime.now(timezone.utc) - last.timestamp).total_seconds() * 1000

    def total_duration_ms(self) -> float | None:
        if not self.history:
            return None
        start = self.history[0].timestamp
        end = self.history[-1].timestamp if self.is_terminal else datetime.now(timezone.utc)
        return (end - start).total_seconds() * 1000