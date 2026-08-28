"""Tests for build profiles, the unified kernel adapter, and the profiler."""

from __future__ import annotations

import pytest

from metainfer.tasks.opt_operator.orchestrator.build import (
    BuildError,
    BuildProfile,
    build_hip,
    build_kernel,
    build_triton,
)
from metainfer.tasks.opt_operator.orchestrator.contract import OperatorContract
from metainfer.tasks.opt_operator.orchestrator.gpu_pool import GpuPool
from metainfer.tasks.opt_operator.orchestrator.kernel_adapter import (
    AdapterError,
    KernelAdapter,
    make_adapter,
)
from metainfer.tasks.opt_operator.orchestrator.profiler import (
    ProfilerError,
    PerfResult,
    profile_case,
    profile_cases,
)
from metainfer.tasks.opt_operator.tests._helpers import FakeExecutor, _fill, contract_dict

from .test_gpu_pool import FakeLease

SRC = "kernel source\n"


def make_contract(**mut):
    return OperatorContract.load(contract_dict(**mut))


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def test_build_profile_defaults():
    hip = BuildProfile.default_for("hip")
    assert hip.language == "hip" and hip.arch == "gfx928"
    triton = BuildProfile.default_for("triton")
    assert triton.language == "triton"
    with pytest.raises(BuildError):
        BuildProfile.default_for("cuda")


def test_build_profile_rejects_bad_language():
    with pytest.raises(BuildError):
        BuildProfile(language="cuda")


def test_build_kernel_dispatches_and_stages_source(tmp_path):
    c = make_contract()
    res = build_kernel(SRC, c, tmp_path, BuildProfile.default_for("triton"))
    assert res.ok and res.language == "triton"
    assert (tmp_path / "src" / f"{c.entrypoint}.py").exists()


def test_build_hip_route(tmp_path):
    c = make_contract()
    res = build_hip(SRC, c, tmp_path)
    assert res.language == "hip"
    assert (tmp_path / "src" / f"{c.entrypoint}.hip").exists()


def test_build_injected_compiler_called(tmp_path):
    c = make_contract()
    calls = []

    def fake_compiler(profile, source, contract, kernel_dir):
        calls.append((profile.language, contract.name))
        from metainfer.tasks.opt_operator.orchestrator.build import BuildResult
        return BuildResult(language=profile.language, artifact="fake", workspace_dir=kernel_dir,
                           metadata={"arch": profile.arch})

    res = build_triton(SRC, c, tmp_path, compiler=fake_compiler)
    assert res.artifact == "fake"
    assert calls == [("triton", "RMSNorm")]


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #

def test_adapter_rejects_bad_language():
    with pytest.raises(AdapterError):
        make_adapter("cuda", runner=lambda *a: {})


def test_adapter_runs_case(tmp_path):
    c = make_contract(shapes={"B": 1, "S": 8, "H": 4})
    case = c.generate_cases()[0]
    build = build_triton(SRC, c, tmp_path)

    def runner(build, contract, case, inputs):
        return {t.name: _fill(t.resolved_shape(case.dims)) for t in contract.outputs}

    adapter = make_adapter("triton", runner)
    out = adapter.run_case(build, c, case, {})
    assert list(out) == ["Y"]
    from metainfer.tasks.opt_operator.orchestrator._compare import shape_of
    assert shape_of(out["Y"]) == [1, 8, 4]


def test_adapter_refuses_failed_build(tmp_path):
    c = make_contract(shapes={"B": 1, "S": 8, "H": 4})
    case = c.generate_cases()[0]
    from metainfer.tasks.opt_operator.orchestrator.build import BuildResult
    bad = BuildResult(language="triton", artifact="x", workspace_dir=tmp_path, ok=False, error="boom")
    adapter = make_adapter("triton", runner=lambda *a: {})
    with pytest.raises(AdapterError):
        adapter.run_case(bad, c, case, {})


def test_adapter_requires_dict_output(tmp_path):
    c = make_contract(shapes={"B": 1, "S": 8, "H": 4})
    case = c.generate_cases()[0]
    build = build_triton(SRC, c, tmp_path)
    adapter = make_adapter("triton", runner=lambda *a: [1, 2, 3])
    with pytest.raises(AdapterError):
        adapter.run_case(build, c, case, {})


# --------------------------------------------------------------------------- #
# Profiler (via GPU pool + injected timer)
# --------------------------------------------------------------------------- #

def _pool_for(fake):
    return GpuPool(node_id="node", holder="orch",
                   discover=lambda: {i: {} for i in (fake.free + fake.held)},
                   acquire=fake.acquire, release=fake.release,
                   poll_s=0.005, slot_deadline_s=0.001)


def test_profile_case_measures_and_releases(tmp_path):
    c = make_contract(shapes={"B": 1, "S": 8, "H": 4})
    case = c.generate_cases()[0]
    build = build_triton(SRC, c, tmp_path)
    fake = FakeLease(free=[0], held=[])
    pool = _pool_for(fake)

    result = profile_case(pool, c, case, build, job_id="j", reps=5,
                          timer=lambda b, ct, cs, lease, reps: 1234.5)
    assert result.case_id == case.id
    assert result.latency_ns == 1234.5
    # slot released back
    assert 0 in fake.free


def test_profile_cases_all(tmp_path):
    c = make_contract(shapes={"B": 1, "S": [8, 16], "H": 4})
    build = build_triton(SRC, c, tmp_path)
    fake = FakeLease(free=[0], held=[])
    pool = _pool_for(fake)
    results = profile_cases(pool, c, build, job_id="j",
                            timer=lambda b, ct, cs, lease, reps: 10.0)
    assert len(results) == 2
    assert all(r.latency_ns == 10.0 for r in results.values())


def test_default_timer_raises_without_env(tmp_path):
    c = make_contract(shapes={"B": 1, "S": 8, "H": 4})
    case = c.generate_cases()[0]
    build = build_triton(SRC, c, tmp_path)
    fake = FakeLease(free=[0], held=[])
    pool = _pool_for(fake)
    with pytest.raises(ProfilerError):
        profile_case(pool, c, case, build, job_id="j")
