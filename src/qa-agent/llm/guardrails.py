"""
Two independent safety concerns bundled under one module because both
guard what crosses the LLM boundary:

1. **PII redaction** — scrubs likely personal data out of text before it
   is sent to an LLM provider (story narratives can contain real
   customer data pasted from a support ticket) and before it's logged.
2. **Action allowlisting** — a second, LLM-specific safety check on top
   of `planning/plan_validator.py`'s guardrails: this module's
   `is_action_permitted` is the primitive both `plan_validator.py` and
   any other LLM-output consumer (e.g. a future chat-driven "run this
   ad hoc action" feature) can call, so the allowlist logic exists in
   exactly one place rather than being re-implemented per caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings
from qa_agent.planning.step_schema import ActionType, DestructiveCategory

logger = get_logger(__name__)

# --- PII redaction -----------------------------------------------------

_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
# Matches common card number groupings (13-19 digits, optionally
# separated by spaces/dashes in groups of 4) without validating a Luhn
# checksum — over-redacting a false positive is far cheaper than leaking
# a real card number.
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IP_ADDRESS_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

_REDACTION_RULES: list[tuple[str, re.Pattern]] = [
    ("EMAIL", _EMAIL_PATTERN),
    ("SSN", _SSN_PATTERN),
    ("CREDIT_CARD", _CREDIT_CARD_PATTERN),
    ("PHONE", _PHONE_PATTERN),
    ("IP_ADDRESS", _IP_ADDRESS_PATTERN),
]


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    redaction_counts: dict[str, int]

    @property
    def had_redactions(self) -> bool:
        return any(count > 0 for count in self.redaction_counts.values())


def redact_pii(text: str) -> RedactionResult:
    """
    Replaces likely PII with a labeled placeholder (e.g. `<EMAIL>`) so
    the LLM still sees that *something* was there (useful for context —
    "the user's <EMAIL> was rejected") without seeing the actual value.

    Order matters: credit-card-like digit sequences are checked before
    phone numbers, since an unformatted long digit run could otherwise
    be partially matched by the phone pattern first.
    """
    redacted = text
    counts: dict[str, int] = {}
    for label, pattern in _REDACTION_RULES:
        redacted, count = pattern.subn(f"<{label}>", redacted)
        counts[label] = count

    total = sum(counts.values())
    if total > 0:
        logger.info("Redacted %d PII match(es) before LLM call: %s", total, counts)

    return RedactionResult(redacted_text=redacted, redaction_counts=counts)


# --- Action allowlisting -------------------------------------------------

# Actions that are never permitted regardless of allowlist configuration
# — there is currently no such action in `ActionType`, but this exists as
# an explicit, always-checked-first hard stop so adding a genuinely
# dangerous future action type (e.g. arbitrary script execution) has an
# obvious place to be permanently blocked, not just allowlist-gated.
_HARD_BLOCKED_ACTIONS: frozenset[ActionType] = frozenset()


class ActionNotPermittedError(Exception):
    def __init__(self, action: ActionType, category: DestructiveCategory, reason: str) -> None:
        self.action = action
        self.category = category
        super().__init__(
            f"Action {action.value!r} (destructive_category={category.value!r}) is not "
            f"permitted: {reason}"
        )


def is_action_permitted(
    action: ActionType,
    destructive_category: DestructiveCategory,
    settings: Settings | None = None,
) -> tuple[bool, str]:
    """
    Returns `(permitted, reason)`. This does NOT replace
    `planning/plan_validator.py`'s human-approval routing for destructive
    steps — that remains the authoritative gate for "should a human sign
    off on this" — this function answers a narrower question: "is this
    action even eligible to be considered, ever, for this deployment,"
    which is checked before a plan reaches the validator at all (e.g. a
    deployment that wants to categorically forbid file uploads regardless
    of destructiveness classification).
    """
    settings = settings or get_settings()

    if action in _HARD_BLOCKED_ACTIONS:
        return False, f"{action.value!r} is permanently blocked and cannot be enabled by any configuration."

    if destructive_category == DestructiveCategory.NONE:
        return True, "non-destructive action."

    allowlist = set(settings.destructive_action_allowlist)
    if destructive_category.value in allowlist:
        return True, f"destructive_category {destructive_category.value!r} is in the configured allowlist."

    if settings.require_human_approval_for_destructive_actions:
        return True, (
            f"destructive_category {destructive_category.value!r} is not allowlisted, but "
            "will be routed for human approval rather than blocked outright."
        )

    return False, (
        f"destructive_category {destructive_category.value!r} is not allowlisted and "
        "human-approval routing is disabled for this deployment."
    )


def enforce_action_permitted(
    action: ActionType,
    destructive_category: DestructiveCategory,
    settings: Settings | None = None,
) -> None:
    """Raises `ActionNotPermittedError` instead of returning a tuple, for call sites that want a hard stop rather than a boolean to branch on."""
    permitted, reason = is_action_permitted(action, destructive_category, settings)
    if not permitted:
        raise ActionNotPermittedError(action, destructive_category, reason)