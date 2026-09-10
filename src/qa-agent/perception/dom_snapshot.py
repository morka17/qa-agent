"""
Extracts a pruned, structured snapshot of a page's interactive and
text-bearing elements.

Playwright's accessibility-tree API has been deprecated across recent
versions in favor of ad-hoc DOM inspection, so this module walks the live
DOM directly via a single `page.evaluate()` call rather than depending on
`page.accessibility.snapshot()`. The JS walker computes an approximate
ARIA role and accessible name for every visible, interactive-or-labeled
element using the same resolution order a real assistive-technology
implementation would (explicit role/aria-label first, then implicit
role-from-tag, then visible text) — the same signal `role_based.py` and
`text_based.py` build their scoring on.

`Page` is expressed as a local `Protocol` (not a Playwright import) so
this module — and everything downstream of it — can be unit-tested with
a lightweight fake rather than a real browser.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from qa_agent.config.logging_config import get_logger

logger = get_logger(__name__)


class Page(Protocol):
    """The minimal Playwright `Page` surface this module depends on."""

    @property
    def url(self) -> str: ...

    async def title(self) -> str: ...

    async def evaluate(self, expression: str) -> Any: ...


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass(frozen=True)
class DomElementNode:
    """
    A single visible, interactive-or-labeled element as extracted from
    the live DOM. `index` is stable for the lifetime of one snapshot and
    is what `element_resolver.py` and its strategies reference — it is
    NOT stable across snapshots/page states, so callers must re-resolve
    (or validate via `selector_cache.py`) after any navigation or DOM
    mutation.
    """

    index: int
    tag: str
    role: str | None
    accessible_name: str | None
    text: str | None
    value: str | None
    attributes: dict[str, str]
    dom_path: str  # a best-effort, execution-time-usable CSS selector path
    bounding_box: BoundingBox | None
    is_visible: bool
    is_enabled: bool

    def searchable_text(self) -> str:
        """All the text signal available for this element, for fuzzy matching."""
        parts = [self.accessible_name, self.text, self.value, self.attributes.get("placeholder")]
        return " ".join(p for p in parts if p).strip()


@dataclass
class DomSnapshot:
    url: str
    title: str
    elements: list[DomElementNode]
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def page_signature(self) -> str:
        """
        A stable fingerprint of this page's structure, used as the cache
        key namespace in `selector_cache.py` and as the comparison basis
        in `page_state_diff.py`. Two snapshots of "the same page" in a
        meaningful sense (same URL, same rough element structure) should
        produce the same signature even if dynamic content (timestamps,
        counts) differs slightly.
        """
        structural_tokens = sorted(
            f"{el.tag}:{el.role or ''}" for el in self.elements
        )
        digest_input = self.url.split("?")[0] + "|" + "|".join(structural_tokens)
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]

    def element_by_index(self, index: int) -> DomElementNode | None:
        return next((e for e in self.elements if e.index == index), None)

    def visible_elements(self) -> list[DomElementNode]:
        return [e for e in self.elements if e.is_visible]


# Walks the live DOM collecting every element that is either natively
# interactive, has an explicit interactive ARIA role, or carries
# meaningful visible text — then computes a best-effort accessible name
# and a short, execution-time-usable CSS path for each. Deliberately
# framework-agnostic (no dependency on a specific app's markup
# conventions) since Sentinel must work against arbitrary target apps.
_SNAPSHOT_JS = r"""
() => {
  const INTERACTIVE_TAGS = new Set([
    'a', 'button', 'input', 'select', 'textarea', 'option', 'label'
  ]);
  const IMPLICIT_ROLES = {
    a: 'link', button: 'button', input: 'textbox', select: 'combobox',
    textarea: 'textbox', option: 'option', img: 'img',
    h1: 'heading', h2: 'heading', h3: 'heading', h4: 'heading',
    h5: 'heading', h6: 'heading',
  };
  const INPUT_ROLE_OVERRIDES = {
    checkbox: 'checkbox', radio: 'radio', submit: 'button',
    button: 'button', reset: 'button', range: 'slider',
  };

  function isVisible(el) {
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
      return false;
    }
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function accessibleName(el) {
    const ariaLabel = el.getAttribute('aria-label');
    if (ariaLabel) return ariaLabel.trim();

    const labelledBy = el.getAttribute('aria-labelledby');
    if (labelledBy) {
      const labelText = labelledBy
        .split(/\s+/)
        .map((id) => document.getElementById(id)?.innerText || '')
        .join(' ')
        .trim();
      if (labelText) return labelText;
    }

    if (el.id) {
      const forLabel = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (forLabel && forLabel.innerText.trim()) return forLabel.innerText.trim();
    }

    const closestLabel = el.closest('label');
    if (closestLabel && closestLabel.innerText.trim()) return closestLabel.innerText.trim();

    if (el.tagName === 'IMG' && el.getAttribute('alt')) return el.getAttribute('alt').trim();

    const title = el.getAttribute('title');
    if (title) return title.trim();

    const placeholder = el.getAttribute('placeholder');
    if (placeholder) return placeholder.trim();

    const text = (el.innerText || '').trim();
    if (text && text.length <= 120) return text;

    return null;
  }

  function computeRole(el) {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'input') {
      const type = (el.getAttribute('type') || 'text').toLowerCase();
      return INPUT_ROLE_OVERRIDES[type] || 'textbox';
    }
    return IMPLICIT_ROLES[tag] || null;
  }

  function cssPath(el) {
    if (el.id) return `#${CSS.escape(el.id)}`;
    const parts = [];
    let node = el;
    let depth = 0;
    while (node && node.nodeType === 1 && depth < 6) {
      let selector = node.tagName.toLowerCase();
      if (node.parentElement) {
        const siblings = Array.from(node.parentElement.children).filter(
          (c) => c.tagName === node.tagName
        );
        if (siblings.length > 1) {
          selector += `:nth-of-type(${siblings.indexOf(node) + 1})`;
        }
      }
      parts.unshift(selector);
      node = node.parentElement;
      depth += 1;
    }
    return parts.join(' > ');
  }

  const candidates = new Set();
  document.querySelectorAll('body, body *').forEach((el) => {
    const tag = el.tagName.toLowerCase();
    const role = computeRole(el);
    const hasDirectText = Array.from(el.childNodes).some(
      (n) => n.nodeType === 3 && n.textContent.trim().length > 0
    );
    if (INTERACTIVE_TAGS.has(tag) || (role && role !== 'generic') || hasDirectText) {
      candidates.add(el);
    }
  });

  const results = [];
  let index = 0;
  candidates.forEach((el) => {
    const rect = el.getBoundingClientRect();
    const attrs = {};
    for (const attr of el.attributes) {
      if (['id', 'class', 'name', 'type', 'placeholder', 'data-testid', 'href'].includes(attr.name)) {
        attrs[attr.name] = attr.value;
      }
    }
    results.push({
      index: index++,
      tag: el.tagName.toLowerCase(),
      role: computeRole(el),
      accessible_name: accessibleName(el),
      text: (el.innerText || '').trim().slice(0, 200) || null,
      value: 'value' in el ? (el.value || null) : null,
      attributes: attrs,
      dom_path: cssPath(el),
      bounding_box: rect.width > 0 && rect.height > 0
        ? { x: rect.x, y: rect.y, width: rect.width, height: rect.height }
        : null,
      is_visible: isVisible(el),
      is_enabled: !el.disabled,
    });
  });

  return { url: window.location.href, title: document.title, elements: results };
}
"""


class DomSnapshotError(Exception):
    """Raised when a page snapshot cannot be captured or parsed."""


class DomSnapshotExtractor:
    """
    Captures a `DomSnapshot` from a live Playwright page.

    Example:
        extractor = DomSnapshotExtractor()
        snapshot = await extractor.capture(page)
    """

    async def capture(self, page: Page) -> DomSnapshot:
        try:
            raw = await page.evaluate(_SNAPSHOT_JS)
        except Exception as exc:  # noqa: BLE001 - surface as a domain-specific error
            raise DomSnapshotError(f"Failed to evaluate DOM snapshot script: {exc}") from exc

        return self._parse_raw_snapshot(raw)

    @staticmethod
    def _parse_raw_snapshot(raw: dict[str, Any]) -> DomSnapshot:
        elements: list[DomElementNode] = []
        for raw_el in raw.get("elements", []):
            bbox_raw = raw_el.get("bounding_box")
            bbox = (
                BoundingBox(
                    x=bbox_raw["x"], y=bbox_raw["y"], width=bbox_raw["width"], height=bbox_raw["height"]
                )
                if bbox_raw
                else None
            )
            elements.append(
                DomElementNode(
                    index=raw_el["index"],
                    tag=raw_el["tag"],
                    role=raw_el.get("role"),
                    accessible_name=raw_el.get("accessible_name"),
                    text=raw_el.get("text"),
                    value=raw_el.get("value"),
                    attributes=raw_el.get("attributes", {}),
                    dom_path=raw_el["dom_path"],
                    bounding_box=bbox,
                    is_visible=bool(raw_el.get("is_visible", False)),
                    is_enabled=bool(raw_el.get("is_enabled", True)),
                )
            )

        snapshot = DomSnapshot(url=raw["url"], title=raw.get("title", ""), elements=elements)
        logger.debug(
            "Captured DOM snapshot for %s: %d element(s), signature=%s",
            snapshot.url,
            len(snapshot.elements),
            snapshot.page_signature,
        )
        return snapshot


def snapshot_to_json(snapshot: DomSnapshot) -> str:
    """Serialize a snapshot for logging/debugging/trace attachment."""
    return json.dumps(
        {
            "url": snapshot.url,
            "title": snapshot.title,
            "captured_at": snapshot.captured_at.isoformat(),
            "page_signature": snapshot.page_signature,
            "elements": [
                {
                    "index": e.index,
                    "tag": e.tag,
                    "role": e.role,
                    "accessible_name": e.accessible_name,
                    "text": e.text,
                    "is_visible": e.is_visible,
                    "is_enabled": e.is_enabled,
                }
                for e in snapshot.elements
            ],
        },
        indent=2,
    )