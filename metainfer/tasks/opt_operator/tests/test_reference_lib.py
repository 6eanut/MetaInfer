"""Tests for the reference library: review gate, atomic admission, resolution order."""

from __future__ import annotations

import json

import pytest

from metainfer.tasks.opt_operator.orchestrator.contract import OperatorContract
from metainfer.tasks.opt_operator.orchestrator.oracle import freeze_reference
from metainfer.tasks.opt_operator.orchestrator.reference_lib import (
    ReferenceLibError,
    ReferenceLibrary,
    ReviewReport,
    make_reference_entry,
    op_id_for,
    review_reference,
)
from metainfer.tasks.opt_operator.tests._helpers import FakeExecutor, contract_dict

SRC = "def forward(**tensors):\n    return {k: v for k, v in tensors.items()}\n"


def make_contract(**mut):
    return OperatorContract.load(contract_dict(**mut))


def test_op_id_slug():
    assert op_id_for(make_contract()) == "rmsnorm"
    assert op_id_for(make_contract(name=" My Op 2 ")) == "my_op_2"


# --------------------------------------------------------------------------- #
# Review gate
# --------------------------------------------------------------------------- #

def test_review_passes_for_good_reference():
    report = review_reference(make_contract(shapes={"B": 1, "S": [8, 16], "H": [4, 8]}),
                              FakeExecutor(), SRC)
    assert report.passed, report.as_dict()


def test_review_fails_nondeterministic():
    report = review_reference(make_contract(shapes={"B": 1, "S": 8, "H": 4}),
                              FakeExecutor(nondeterministic=True), SRC)
    assert not report.passed
    assert not report.check("deterministic").passed


def test_review_fails_wrong_shape():
    report = review_reference(make_contract(shapes={"B": 1, "S": 8, "H": 4}),
                              FakeExecutor(wrong_shape=True), SRC)
    assert not report.passed
    assert not report.check("shapes_match").passed


def test_review_fails_nonfinite():
    report = review_reference(make_contract(shapes={"B": 1, "S": 8, "H": 4}),
                              FakeExecutor(nonfinite=True), SRC)
    assert not report.passed
    assert not report.check("finite").passed


def test_review_fails_crash():
    report = review_reference(make_contract(shapes={"B": 1, "S": 8, "H": 4}),
                              FakeExecutor(crash=True), SRC)
    assert not report.passed


# --------------------------------------------------------------------------- #
# Admission + lookup
# --------------------------------------------------------------------------- #

def test_admit_then_find(tmp_path):
    lib = ReferenceLibrary(tmp_path / "ref_lib")
    entry = make_reference_entry(make_contract(), SRC)
    report = ReviewReport(True, [])
    assert lib.admit(entry, report) is True

    found = lib.find(entry.op_id)
    assert found is not None
    assert found.digest == entry.digest
    assert found.reference_source == SRC
    assert lib.list_op_ids() == [entry.op_id]


def test_admit_refuses_failed_review(tmp_path):
    lib = ReferenceLibrary(tmp_path / "ref_lib")
    entry = make_reference_entry(make_contract(), SRC)
    with pytest.raises(ReferenceLibError):
        lib.admit(entry, ReviewReport(False, []))


# --------------------------------------------------------------------------- #
# Resolution order (user -> library -> generated)
# --------------------------------------------------------------------------- #

def test_resolve_user_wins(tmp_path):
    lib = ReferenceLibrary(tmp_path / "ref_lib")
    contract = make_contract()
    oracle = lib.resolve(contract, FakeExecutor(),
                         system_oracle_dir=tmp_path / "sys",
                         run_id="r1",
                         user_reference="# user\n" + SRC)
    assert oracle.origin == "user"
    assert oracle.reference_source.startswith("# user")


def test_resolve_library_hit_when_no_user(tmp_path):
    lib = ReferenceLibrary(tmp_path / "ref_lib")
    contract = make_contract()
    # First: admit a library entry for this op.
    entry = make_reference_entry(contract, SRC)
    lib.admit(entry, ReviewReport(True, []))

    oracle = lib.resolve(contract, FakeExecutor(),
                         system_oracle_dir=tmp_path / "sys",
                         run_id="r1")
    assert oracle.origin == "library"
    assert oracle.digest == entry.digest


def test_resolve_generated_pass_admits(tmp_path):
    lib = ReferenceLibrary(tmp_path / "ref_lib")
    contract = make_contract()
    oracle = lib.resolve(contract, FakeExecutor(),
                         system_oracle_dir=tmp_path / "sys",
                         run_id="r1",
                         generated_reference=SRC)
    assert oracle.origin == "generated"
    # It should have been admitted to the library.
    assert lib.find(oracle.op_id) is not None


def test_resolve_generated_fail_not_admitted(tmp_path):
    lib = ReferenceLibrary(tmp_path / "ref_lib")
    contract = make_contract()
    oracle = lib.resolve(contract, FakeExecutor(nondeterministic=True),
                         system_oracle_dir=tmp_path / "sys",
                         run_id="r1",
                         generated_reference=SRC)
    assert oracle.origin == "generated_unadmitted"
    assert lib.find(oracle.op_id) is None


def test_resolve_no_source_raises(tmp_path):
    lib = ReferenceLibrary(tmp_path / "ref_lib")
    with pytest.raises(ReferenceLibError):
        lib.resolve(make_contract(), FakeExecutor(),
                    system_oracle_dir=tmp_path / "sys", run_id="r1")


def test_resolve_frozen_then_reload(tmp_path):
    lib = ReferenceLibrary(tmp_path / "ref_lib")
    contract = make_contract()
    oracle = lib.resolve(contract, FakeExecutor(),
                         system_oracle_dir=tmp_path / "sys",
                         run_id="r1",
                         user_reference=SRC)
    reloaded = lib.reload(tmp_path / "sys", "r1")
    assert reloaded is not None
    assert reloaded.digest == oracle.digest
    assert reloaded.origin == "user"


# --------------------------------------------------------------------------- #
# SSOT / integrity: a tampered library entry is not returned as a hit
# --------------------------------------------------------------------------- #

def test_find_returns_none_for_tampered_entry(tmp_path):
    lib = ReferenceLibrary(tmp_path / "ref_lib")
    entry = make_reference_entry(make_contract(), SRC)
    lib.admit(entry, ReviewReport(True, []))

    # Tamper the on-disk reference.py so its digest no longer matches meta.json.
    (tmp_path / "ref_lib" / entry.op_id / "reference.py").write_text(
        SRC + "# tampered", encoding="utf-8")
    assert lib.find(entry.op_id) is None
