"""SystemBuilder / BuildProfile — the HIP (gfx928) and Triton build routes.

A ``BuildProfile`` captures the toolchain target for one language. ``build_kernel``
dispatches a kernel source to the matching route, which writes the source into the
run workspace and (where a compiler is wired up) produces a :class:`BuildResult`
pointing at the compiled artifact.

The heavy compilation itself (``hipcc`` for HIP, Triton's JIT for Triton) is
environment-bound and injected as a ``compiler`` callable so the pipeline can be
unit-tested without a real toolchain. The default compiler is a no-op that just
stages the source — real compilation is wired by the orchestrator when a K100 /
ROCm environment is present.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from metainfer.cluster.fs_primitives import atomic_write_text

from .contract import OperatorContract


class BuildError(ValueError):
    """Kernel could not be built (bad language / compile failure)."""


LANGUAGES = ("hip", "triton")


def kernel_digest(source: str, language: str = "") -> str:
    """Content digest identifying a kernel source (used for the lineage ledger)."""
    blob = f"{language}\n{source}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BuildProfile:
    language: str
    arch: str = "gfx928"          # HIP: target DCU arch; Triton: unused
    note: str = ""

    def __post_init__(self) -> None:
        if self.language not in LANGUAGES:
            raise BuildError(f"language must be one of {LANGUAGES}, got {self.language!r}")

    @classmethod
    def default_for(cls, language: str) -> "BuildProfile":
        if language == "hip":
            return cls(language="hip", arch="gfx928", note="K100 DCU / DTK")
        if language == "triton":
            return cls(language="triton", arch="gfx928",
                       note="system Triton + ROCm/torch")
        raise BuildError(f"unknown language {language!r}")


@dataclass(frozen=True)
class BuildResult:
    language: str
    artifact: str                 # path / module id of the built product
    workspace_dir: Path
    ok: bool = True
    error: str = ""
    digest: str = ""              # kernel_digest(source, language)
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "artifact": self.artifact,
            "workspace_dir": str(self.workspace_dir),
            "ok": self.ok,
            "error": self.error,
            "digest": self.digest,
            "metadata": self.metadata or {},
        }


# A compiler is callable(profile, source, contract, kernel_dir) -> BuildResult
Compiler = Callable[[BuildProfile, str, OperatorContract, Path], BuildResult]


def _stage_source(language: str, source: str, contract: OperatorContract,
                  workspace: Path) -> Path:
    ext = "hip" if language == "hip" else "py"
    fname = f"{contract.entrypoint or contract.name}.{ext}"
    target = Path(workspace) / "src" / fname
    atomic_write_text(target, source)
    return target


def _default_compiler(profile: BuildProfile, source: str,
                      contract: OperatorContract, kernel_dir: Path) -> BuildResult:
    """Stub compiler — stages the source; real toolchain wired by orchestrator."""
    target = _stage_source(profile.language, source, contract, kernel_dir)
    return BuildResult(language=profile.language, artifact=str(target),
                       workspace_dir=kernel_dir, ok=True,
                       digest=kernel_digest(source, profile.language),
                       metadata={"staged": True, "arch": profile.arch})


def build_kernel(
    source: str,
    contract: OperatorContract,
    workspace: Path,
    profile: BuildProfile,
    compiler: Optional[Compiler] = None,
) -> BuildResult:
    """Dispatch a kernel source to the correct route and build it."""
    comp = compiler or _default_compiler
    return comp(profile, source, contract, Path(workspace))


def build_hip(source: str, contract: OperatorContract, workspace: Path,
              profile: Optional[BuildProfile] = None,
              compiler: Optional[Compiler] = None) -> BuildResult:
    profile = profile or BuildProfile.default_for("hip")
    return build_kernel(source, contract, workspace, profile, compiler)


def build_triton(source: str, contract: OperatorContract, workspace: Path,
                 profile: Optional[BuildProfile] = None,
                 compiler: Optional[Compiler] = None) -> BuildResult:
    profile = profile or BuildProfile.default_for("triton")
    return build_kernel(source, contract, workspace, profile, compiler)


__all__ = [
    "BuildError", "BuildProfile", "BuildResult", "kernel_digest",
    "build_kernel", "build_hip", "build_triton", "LANGUAGES",
]
