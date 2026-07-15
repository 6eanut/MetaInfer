"""Oracle harness import integrity tests.

The C_test phase runs the gen_infer_framework oracle harnesses which
execute model inference in a subprocess.  A broken import path inside
an oracle helper function (e.g. ``from ....gpu_preflight import
preflight_gpu`` → ``ModuleNotFoundError: No module named
'metainfer.tasks.gpu_preflight'``) will only surface at runtime —
the harness file itself imports cleanly because the offending line
sits inside a function body.

These tests exercise every delayed import inside the oracle harness
functions to catch wrong relative-import paths before they reach
production.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest


def _extract_function_imports(source: str, func_name: str) -> list[str]:
    """Parse *source* and return every ``from X import Y`` statement
    found directly inside the body of *func_name*.  Only top-level
    function-body imports are returned (not nested inside if/for/with).
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                imports: list[str] = []
                for stmt in node.body:
                    if isinstance(stmt, ast.ImportFrom):
                        imports.append(
                            f"from {stmt.module or ''} import "
                            + ", ".join(a.asname or a.name for a in stmt.names)
                        )
                    elif isinstance(stmt, ast.Import):
                        imports.append(
                            "import "
                            + ", ".join(a.asname or a.name for a in stmt.names)
                        )
                return imports
    raise ValueError(f"function {func_name!r} not found in source")


# --------------------------------------------------------------------------- #
# Known oracle harness functions that contain lazy (function-body) imports.
# Each entry: (file_stem, function_name, expected_import_module)
# --------------------------------------------------------------------------- #
_HARNESS_IMPORTS = [
    # correctness.py → _start_server
    (
        "correctness.py",
        "_start_server",
        "metainfer.orchestrator.gpu_preflight",
    ),
    # perf.py → _start_server
    (
        "perf.py",
        "_start_server",
        "metainfer.orchestrator.gpu_preflight",
    ),
]


@pytest.mark.parametrize("file_stem,func_name,expected_module", _HARNESS_IMPORTS)
def test_oracle_harness_imports_resolve(file_stem, func_name, expected_module):
    """Every lazy import inside oracle harness functions must resolve.

    The body of each harness function is parsed via AST; any
    ``from <module> import …`` found inside triggers a real
    ``importlib.import_module`` to verify the module exists.
    """
    import importlib

    oracles_dir = (
        Path(__file__).resolve().parent.parent
        / "orchestrator" / "oracles"
    )
    source = (oracles_dir / file_stem).read_text(encoding="utf-8")

    func_imports = _extract_function_imports(source, func_name)
    if not func_imports:
        pytest.skip(f"{file_stem}::{func_name} has no function-body imports")

    for imp_line in func_imports:
        # Extract the module part from "from X import Y"
        m = re.match(r"from\s+(\S+)\s+import", imp_line)
        if not m:
            continue
        module_name = m.group(1)
        # Resolve relative imports only for those inside the oracles subpackage.
        if module_name.startswith("."):
            resolved = _resolve_relative(
                module_name,
                f"metainfer.tasks.gen_infer_framework.orchestrator.oracles.{file_stem.replace('.py', '')}",
            )
        else:
            resolved = module_name
        try:
            importlib.import_module(resolved)
        except ImportError as exc:
            pytest.fail(
                f"In {file_stem}::{func_name}: {imp_line!r} "
                f"→ {resolved} → ImportError: {exc}"
            )


def _resolve_relative(rel_dots: str, caller_module: str) -> str:
    """Resolve a dotted relative import against *caller_module*."""
    level = 0
    for ch in rel_dots:
        if ch == ".":
            level += 1
        else:
            break
    parts = caller_module.split(".")
    if level > len(parts):
        raise ValueError(
            f"Too many leading dots ({level}) for {caller_module!r}"
        )
    base = parts[: -(level - 1)] if level > 1 else parts
    suffix = rel_dots[level:]
    return ".".join(base + ([suffix] if suffix else []))


# --------------------------------------------------------------------------- #
# Gpu preflight import — direct sanity check
# --------------------------------------------------------------------------- #
def test_gpu_preflight_importable():
    """The gpu_preflight module must be importable from its canonical path."""
    from metainfer.orchestrator.gpu_preflight import preflight_gpu
    assert callable(preflight_gpu)


def test_oracle_files_import_cleanly():
    """The oracle module files themselves must not raise ImportError.

    This catches module-level broken imports (not the function-body
    kind tested above, which are covered by the parametrized test).
    """
    from metainfer.tasks.gen_infer_framework.orchestrator.oracles import (
        correctness,
        perf,
    )
    assert correctness.InferFrameworkOracle is not None
    assert perf.PerfOracle is not None
