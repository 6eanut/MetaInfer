"""Adversarial review of the correctness + benchmark harnesses (FR-2/FR-3).

A harness is only trustworthy if we can prove it is *not a rubber stamp* — that
it will actually catch a wrong kernel. This module reviews each harness by
adversarially constructing candidate outputs that are wrong in controlled ways
and asserting the harness flags them. The review conclusions (including the
negative-case evidence) are left on the record so the WebUI can show *why* a
run's correctness verdict is believable.

Because the correctness gate (:func:`conformance.conformance_report`) compares
output tensors, the adversarial proof runs at the **outputs** level — no GPU or
toolchain needed — which keeps this module pure and unit-testable with fake
nested-tensor outputs. The production run constructs the negative cases from the
frozen oracle's own (passing) outputs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .conformance import conformance_report
from .contract import OperatorContract


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NegativeEvidence:
    name: str
    constructed: str            # what the adversarial candidate did
    harness_caught: bool        # did the harness flag it?
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "constructed": self.constructed,
            "harness_caught": self.harness_caught,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReviewCheck:
    name: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class HarnessReview:
    kind: str                       # "correctness" | "benchmark"
    passed: bool
    checks: List[ReviewCheck]
    negative_evidence: List[NegativeEvidence] = field(default_factory=list)
    message: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "passed": self.passed,
            "checks": [c.as_dict() for c in self.checks],
            "negative_evidence": [e.as_dict() for e in self.negative_evidence],
            "message": self.message,
        }


# --------------------------------------------------------------------------- #
# Structural tensor helpers (nested lists / numpy-like arrays)
# --------------------------------------------------------------------------- #

def _is_array(v: Any) -> bool:
    # numpy arrays expose .reshape / .copy; nested lists/tuples don't.
    return hasattr(v, "reshape") and hasattr(v, "copy")


def _bias_one_element(t: Any) -> Any:
    """Return a copy of tensor ``t`` with one element pushed far out of tolerance."""
    if _is_array(t):
        c = t.copy()
        if c.size:
            c.reshape(-1)[0] = c.reshape(-1)[0] + 1000.0
        return c
    # nested list: bias the innermost first leaf
    path = _first_leaf_path(t)
    if not path:
        return copy.deepcopy(t)
    return _mutate_at(t, path, lambda x: x + 1000.0)


def _corrupt_shape(t: Any) -> Any:
    """Return a structurally different (wrong-length) copy of tensor ``t``."""
    if _is_array(t):
        if t.ndim and t.shape[-1] > 1:
            return t[..., :-1]
        return t[..., :1] if t.ndim else t
    # nested list: drop one leaf so the structure no longer matches
    return _drop_leaf(copy.deepcopy(t))


def _first_leaf_path(out_tensor: Any) -> List[int]:
    """A path (list of indices) down to a leaf element of a nested tensor."""
    path: List[int] = []
    cur = out_tensor
    while isinstance(cur, (list, tuple)) and len(cur):
        path.append(0)
        cur = cur[0]
    return path


def _mutate_at(v: Any, path: List[int], op) -> Any:
    """Apply ``op`` to the scalar leaf at ``path`` (coordinates from the top)."""
    if not path:
        return op(v) if not isinstance(v, (list, tuple)) else v
    idx = path[0]
    if isinstance(v, (list, tuple)):
        nxt = list(v)
        nxt[idx] = _mutate_at(nxt[idx], path[1:], op)
        return nxt
    return op(v)


def review_correctness(contract: OperatorContract, oracle_outputs: Dict[str, Dict[str, Any]]) -> HarnessReview:
    """Review the correctness gate against adversarially wrong candidates.

    ``oracle_outputs`` is the frozen oracle's *passing* outputs (case_id ->
    output-name -> tensor). We (a) sanity-check that the gate passes the oracle
    itself, then (b) construct several deliberately-wrong candidates and assert
    the gate catches each one. Only a gate that catches all of them is passed.
    """
    checks: List[ReviewCheck] = []
    evidence: List[NegativeEvidence] = []

    if not oracle_outputs:
        return HarnessReview("correctness", False,
                             [ReviewCheck("oracle_outputs", False,
                                          "no oracle outputs to review")])

    # (a) The gate must accept the oracle as itself (identity is not wrongly
    #     rejected — otherwise the gate is unusable / over-strict).
    identity = conformance_report(contract, oracle_outputs, oracle_outputs)
    checks.append(ReviewCheck(
        "identity_passes", identity.passed,
        "" if identity.passed else "gate rejects the oracle itself"))

    # (b) Adversarial candidates — each constructed to be genuinely wrong.
    perturbations: List[Dict[str, Any]] = _correctness_perturbations(oracle_outputs)
    for p in perturbations:
        caught = not conformance_report(contract, oracle_outputs,
                                        p["candidate"]).passed
        evidence.append(NegativeEvidence(
            name=p["name"], constructed=p["constructed"],
            harness_caught=caught,
            detail="" if caught else "gate FAILED to catch this wrong candidate",
        ))

    checks.append(ReviewCheck(
        "negative_cases_caught", all(e.harness_caught for e in evidence),
        f"{sum(1 for e in evidence if e.harness_caught)}/{len(evidence)} "
        "adversarial wrong candidates caught"))
    passed = identity.passed and all(c.passed for c in checks)
    return HarnessReview("correctness", passed, checks, evidence,
                         message="" if passed else "correctness gate is not trustworthy")


def _correctness_perturbations(oracle_outputs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the set of deliberately-wrong candidate-output mutations."""
    out: List[Dict[str, Any]] = []
    case_ids = list(oracle_outputs.keys())
    if not case_ids:
        return out

    def clone():
        return copy.deepcopy(oracle_outputs)

    # 1. Bias a single element far outside tolerance.
    cand = clone()
    cid = case_ids[0]
    oname = next(iter(cand[cid]))
    cand[cid][oname] = _bias_one_element(cand[cid][oname])
    out.append({"name": "element_bias",
                "constructed": f"added +1000 to one element of {cid}/{oname}",
                "candidate": cand})

    # 2. Scale an entire case's outputs (wrong on that shape only).
    cand = clone()
    bad_cid = case_ids[-1]
    cand[bad_cid] = {k: _scale(v, 100.0) for k, v in cand[bad_cid].items()}
    out.append({"name": "one_case_scaled",
                "constructed": f"scaled all outputs of case {bad_cid} by 100x",
                "candidate": cand})

    # 3. Break the output shape on one case.
    cand = clone()
    cid = case_ids[0]
    oname = next(iter(cand[cid]))
    cand[cid][oname] = _corrupt_shape(cand[cid][oname])
    out.append({"name": "wrong_shape",
                "constructed": f"corrupted the shape of {cid}/{oname}",
                "candidate": cand})

    # 4. Drop an entire case from the candidate.
    if len(case_ids) > 1:
        cand = clone()
        cand.pop(case_ids[-1])
        out.append({"name": "missing_case",
                    "constructed": f"candidate omitted case {case_ids[-1]}",
                    "candidate": cand})
    return out


def _scale(v: Any, s: float) -> Any:
    if isinstance(v, (list, tuple)):
        return [_scale(x, s) for x in v]
    return v * s


def _drop_leaf(v: Any) -> Any:
    """Return a structurally different (shorter / wrong-rank) version of ``v``."""
    if isinstance(v, (list, tuple)):
        if v:
            return _drop_leaf(v[:-1]) if len(v) > 1 else v[0] if isinstance(v[0], list) else []
        return v
    return v  # scalar leaf unchanged


# --------------------------------------------------------------------------- #
# Benchmark review (methodology authority)
# --------------------------------------------------------------------------- #

def review_benchmark(meta: Dict[str, Any],
                     correctness_shape_ids: Optional[List[str]] = None) -> HarnessReview:
    """Review the benchmark harness's authority from its metadata.

    A benchmark verdict is only credible if it used warmup, enough reps, a stable
    statistic, and the same shape set as the correctness harness (so the two
    gates agree on what they are measuring). Each missing/invalid item is a
    failed check, and the conclusion is left on the record.
    """
    checks: List[ReviewCheck] = []
    shapes = meta.get("shape_ids") or []
    checks.append(ReviewCheck("shape_set_present", bool(shapes),
                              f"{len(shapes)} shapes" if shapes else "no shape set"))
    if correctness_shape_ids:
        aligned = list(shapes) == list(correctness_shape_ids)
        checks.append(ReviewCheck(
            "shape_aligned_with_correctness", aligned,
            "" if aligned else "benchmark shape set != correctness shape set"))
    warmup = meta.get("warmup")
    checks.append(ReviewCheck("warmup_specified",
                              isinstance(warmup, int) and warmup >= 0,
                              f"warmup={warmup}"))
    reps = meta.get("reps")
    checks.append(ReviewCheck("reps_specified", isinstance(reps, int) and reps >= 1,
                              f"reps={reps}"))
    statistic = meta.get("statistic")
    checks.append(ReviewCheck("stable_statistic",
                              statistic in ("median", "mean"),
                              f"statistic={statistic}"))
    baseline = meta.get("baseline_digest")
    checks.append(ReviewCheck("baseline_defined", bool(baseline),
                              f"baseline_digest={baseline or 'none'}"))
    passed = all(c.passed for c in checks)
    return HarnessReview("benchmark", passed, checks,
                         message="" if passed else "benchmark harness lacks authority")


__all__ = ["NegativeEvidence", "ReviewCheck", "HarnessReview",
           "review_correctness", "review_benchmark"]
