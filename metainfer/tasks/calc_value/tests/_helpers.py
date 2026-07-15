"""calc_value-specific test helpers.

Lives inside the calc_value task package (not the shared
:mod:`metainfer.testing` package) because it's only useful for tests
that exercise the calc_value orchestrator or web plugin. Other task
packages should ship their own helpers in their own ``tests/`` dir.
"""

from __future__ import annotations

from pathlib import Path


def write_calc_script(
    path: Path,
    *,
    prefill_tflops: float,
    prefill_gb: float,
    decode_tflops: float = 0.0,
    decode_gb: float = 0.0,
    legacy_shape: bool = False,
) -> Path:
    """Write a minimal ``calc.py`` at ``path`` returning the given numbers.

    If ``legacy_shape`` is True, returns the old single-dict shape
    ``{"tflops": ..., "access_gb": ...}`` — useful for testing the
    backward-compat path in ``call_calc``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if legacy_shape:
        content = (
            "def calc(batch_size, seq_len):\n"
            f"    return {{'tflops': {prefill_tflops}, 'access_gb': {prefill_gb}}}\n"
        )
    else:
        content = (
            "def calc(batch_size, seq_len):\n"
            "    return {\n"
            f"        'prefill': {{'tflops': {prefill_tflops}, 'access_gb': {prefill_gb}}},\n"
            f"        'decode':  {{'tflops': {decode_tflops}, 'access_gb': {decode_gb}}},\n"
            "    }\n"
        )
    path.write_text(content, encoding="utf-8")
    return path
