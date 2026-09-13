"""
Loads versioned prompt templates from disk rather than requiring every
LLM-calling module to inline its system prompt as a Python string.

Existing modules (`ingestion/story_parser.py`, `planning/test_planner.py`,
etc.) currently define their prompts as inline `_SYSTEM_PROMPT` constants
— that's fine for prompts that are simple and stable. This loader is the
supported path for prompts a team wants to iterate on independently of a
code deploy (reviewed/edited without touching Python), tracked here as
plain text files so `docs/adr/0003-llm-provider-abstraction.md`'s
"every LLM call is a versioned, auditable prompt" principle has an actual
mechanism behind it for prompts that outgrow an inline string.

Templates use `string.Template`'s `$variable` syntax rather than
`str.format()`'s `{variable}` — deliberately, since prompt text routinely
contains literal `{` `}` (JSON examples, code snippets) that would
otherwise need escaping throughout every template file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

_TEMPLATES_DIR = Path(__file__).parent


class PromptTemplateError(Exception):
    """Raised when a template file is missing or a required variable isn't supplied."""


@lru_cache(maxsize=64)
def _read_template_file(name: str) -> Template:
    path = _TEMPLATES_DIR / f"{name}.txt"
    if not path.exists():
        available = sorted(p.stem for p in _TEMPLATES_DIR.glob("*.txt"))
        raise PromptTemplateError(f"No prompt template named {name!r} found. Available: {available}")
    return Template(path.read_text(encoding="utf-8"))


def load_template(name: str, **variables: str) -> str:
    """
    Load `<name>.txt` from this package and substitute `$variable`
    placeholders with the given keyword arguments.

    Example:
        prompt = load_template("story_parser_system")
        prompt = load_template("bug_title_hint", story_title="Add to cart")

    Raises:
        PromptTemplateError: the named template doesn't exist, or the
            template references a `$variable` not supplied in `variables`
            (safe_substitute is deliberately NOT used — a silently
            unsubstituted placeholder in a prompt sent to an LLM is worse
            than a loud failure at call time).
    """
    template = _read_template_file(name)
    try:
        return template.substitute(**variables)
    except KeyError as exc:
        raise PromptTemplateError(
            f"Template {name!r} references variable {exc} which was not provided."
        ) from exc


def list_templates() -> list[str]:
    return sorted(p.stem for p in _TEMPLATES_DIR.glob("*.txt"))


def clear_template_cache() -> None:
    """Mainly for tests / hot-reload during prompt iteration — production doesn't need this."""
    _read_template_file.cache_clear()