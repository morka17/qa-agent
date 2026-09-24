"""
Compares a current screenshot against a stored baseline to catch visual
regressions that structural assertions can't express — a button that's
technically present and clickable but rendered off-screen, invisible
text (white-on-white), or a broken layout.

Diffing is pixel-level (per-pixel RGB delta) rather than perceptual/SSIM,
which keeps this dependency-light (Pillow only, already used by
`perception/selector_strategies/visual_grounding.py`) at the cost of
being more sensitive to sub-pixel anti-aliasing noise than a perceptual
metric would be — mitigated by `mismatch_ratio` thresholding and an
optional per-pixel tolerance rather than requiring an exact match.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops

from qa_agent.config.logging_config import get_logger

logger = get_logger(__name__)

# Below this fraction of differing pixels, two screenshots are considered
# the same for practical purposes (font rendering/anti-aliasing jitter
# across runs is normal and not a regression).
_DEFAULT_MISMATCH_THRESHOLD = 0.01

# Per-channel RGB delta below which a pixel is not counted as "different"
# at all — filters out 1-2 value anti-aliasing noise.
_DEFAULT_PIXEL_TOLERANCE = 12


class VisualRegressionError(Exception):
    """Raised when a baseline/current image can't be loaded or compared."""


@dataclass(frozen=True)
class VisualDiffResult:
    baseline_size: tuple[int, int]
    current_size: tuple[int, int]
    size_matches: bool
    differing_pixel_count: int
    total_pixel_count: int
    mismatch_ratio: float
    passed: bool
    diff_image_bytes: bytes | None  # visual heat-map of differences, for the filed bug report

    @property
    def mismatch_percent(self) -> float:
        return round(self.mismatch_ratio * 100, 2)


def _load_image(data: bytes | Path, label: str) -> Image.Image:
    try:
        if isinstance(data, Path):
            return Image.open(data).convert("RGB")
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise VisualRegressionError(f"Could not load {label} image: {exc}") from exc


def _build_diff_mask(baseline: Image.Image, current: Image.Image, tolerance: int) -> Image.Image:
    """
    Per-pixel absolute difference, thresholded by `tolerance` so minor
    anti-aliasing noise doesn't register. Returns a single-channel ("L")
    image where non-zero pixels are meaningfully different.
    """
    diff = ImageChops.difference(baseline, current)
    grayscale_diff = diff.convert("L")
    return grayscale_diff.point(lambda p: 255 if p > tolerance else 0)


def _render_heatmap(current: Image.Image, mask: Image.Image) -> bytes:
    """
    Overlay differing regions in red on a copy of the current screenshot,
    producing a human-scannable diff image for the filed bug report.
    """
    overlay = Image.new("RGB", current.size, (255, 0, 0))
    highlighted = Image.composite(overlay, current, mask)
    buffer = io.BytesIO()
    highlighted.save(buffer, format="PNG")
    return buffer.getvalue()


def compare(
    baseline: bytes | Path,
    current: bytes | Path,
    mismatch_threshold: float = _DEFAULT_MISMATCH_THRESHOLD,
    pixel_tolerance: int = _DEFAULT_PIXEL_TOLERANCE,
    generate_diff_image: bool = True,
) -> VisualDiffResult:
    """
    Compare two screenshots and report whether they match within
    tolerance.

    A size mismatch (e.g. a responsive layout shift, a different
    viewport) always fails the comparison outright — pixel diffing two
    differently-sized images isn't meaningful, so `passed` is False and
    `mismatch_ratio` is reported as 1.0 without attempting to align them.
    """
    baseline_img = _load_image(baseline, "baseline")
    current_img = _load_image(current, "current")

    if baseline_img.size != current_img.size:
        logger.warning(
            "Visual regression: size mismatch baseline=%s vs current=%s.",
            baseline_img.size,
            current_img.size,
        )
        return VisualDiffResult(
            baseline_size=baseline_img.size,
            current_size=current_img.size,
            size_matches=False,
            differing_pixel_count=0,
            total_pixel_count=0,
            mismatch_ratio=1.0,
            passed=False,
            diff_image_bytes=None,
        )

    mask = _build_diff_mask(baseline_img, current_img, pixel_tolerance)
    histogram = mask.histogram()
    differing_pixels = histogram[255] if len(histogram) > 255 else 0
    total_pixels = mask.size[0] * mask.size[1]
    mismatch_ratio = differing_pixels / total_pixels if total_pixels else 0.0
    passed = mismatch_ratio <= mismatch_threshold

    diff_bytes = _render_heatmap(current_img, mask) if generate_diff_image and not passed else None

    logger.debug(
        "Visual regression: %.4f%% mismatch (%d/%d px), threshold=%.4f%%, passed=%s.",
        mismatch_ratio * 100,
        differing_pixels,
        total_pixels,
        mismatch_threshold * 100,
        passed,
    )

    return VisualDiffResult(
        baseline_size=baseline_img.size,
        current_size=current_img.size,
        size_matches=True,
        differing_pixel_count=differing_pixels,
        total_pixel_count=total_pixels,
        mismatch_ratio=round(mismatch_ratio, 6),
        passed=passed,
        diff_image_bytes=diff_bytes,
    )


class BaselineStore:
    """
    Minimal filesystem-backed baseline manager. A production deployment
    might swap this for artifact-store-backed storage (S3/GCS), but the
    local-disk default is enough for CI runs where baselines are checked
    into the repo alongside `tests/golden_reports/`.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self._root / f"{safe_key}.png"

    def exists(self, key: str) -> bool:
        return self._path_for(key).exists()

    def load(self, key: str) -> bytes:
        path = self._path_for(key)
        if not path.exists():
            raise VisualRegressionError(f"No baseline stored for key {key!r} at {path}.")
        return path.read_bytes()

    def save(self, key: str, image_bytes: bytes) -> Path:
        path = self._path_for(key)
        path.write_bytes(image_bytes)
        logger.info("Saved baseline %r -> %s.", key, path)
        return path