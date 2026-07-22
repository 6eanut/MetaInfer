"""Phase / Outcome / Transition definitions for port-model.

A six-agent pipeline that ports a model to a target inference framework:

    P1_weight_analysis      — 权重参数分析 Agent
    P2_framework_analysis   — 推理框架分析师 Agent (fan-out, one per ref source)
    P3_architect_review     — 推理框架资深架构师 Agent (汇总 + 可打回 P1)
    P4_minimal_framework    — 精简推理框架编写工程师 Agent
    P5_verify_minimal       — 精简推理框架验证工程师 Agent (失败 → 回 P4)
    P6_port_engine          — 推理引擎移植工程师 Agent (内部调试循环)
    finished

Notable transitions:
  * P3 bounce — the architect may decide the model weight analysis
    diverges too much from the reference framework analysts and ask
    for a redo. Capped at ``MAX_P3_BOUNCE`` (default 2) by the
    pipeline; after the cap P3 must accept and proceed.
  * P5 repair — minimal-framework verification failure routes back to
    P4 with the captured error log; capped at ``MAX_P5_REPAIR`` (3).
  * P6 self-loop — the porting engineer runs its own internal
    debug-loop (similarity comparison against P5 dumps). Each P6
    attempt that needs more work transitions P6 → P6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

Phase = Literal[
    "idle",
    "P1_weight_analysis",
    "P2_framework_analysis",
    "P3_architect_review",
    "P4_minimal_framework",
    "P5_verify_minimal",
    "P6_port_engine",
    "finished",
]

Outcome = Literal[
    "ok",
    "logic_fail",
    "infra_fail",
    "test_fail",     # P5: minimal framework crashed or output not sensible
    "bounce_back",   # P3: send back to P1 for redo
    "needs_repair",  # P6: similarity below threshold, retry within P6
    "aborted",
]

OK = "ok"
LOGIC_FAIL = "logic_fail"
INFRA_FAIL = "infra_fail"
TEST_FAIL = "test_fail"
BOUNCE_BACK = "bounce_back"
NEEDS_REPAIR = "needs_repair"
ABORTED = "aborted"

ALL_OUTCOMES: List[Outcome] = [
    OK, LOGIC_FAIL, INFRA_FAIL, TEST_FAIL, BOUNCE_BACK, NEEDS_REPAIR, ABORTED,
]


@dataclass(frozen=True)
class PhaseMeta:
    id: Phase
    label: str
    description: str = ""
    is_terminal: bool = False


@dataclass(frozen=True)
class Transition:
    from_phase: Phase
    on: Outcome
    to_phase: Phase
    label: str = ""
    consume_iteration: bool = False


PHASES: List[PhaseMeta] = [
    PhaseMeta("idle", "idle", "not started"),
    PhaseMeta(
        "P1_weight_analysis", "1: Weight analysis",
        "权重参数分析 Agent：读取 config.json、量化配置、各权重张量的维度"
        "与命名，落盘为结构化 markdown。",
    ),
    PhaseMeta(
        "P2_framework_analysis", "2: Framework analysts (fan-out)",
        "推理框架分析师 Agent：每个参考实现各启一个，分析模型结构 + 权重加"
        "载/反量化逻辑，标注抽象描述与实际代码位置、算子输入输出形状。",
    ),
    PhaseMeta(
        "P3_architect_review", "3: Architect review",
        "推理框架资深架构师 Agent：交叉对比 P2 多份结果，可二次检索确认，"
        "可打回 P1 重做（最多 2 次）。",
    ),
    PhaseMeta(
        "P4_minimal_framework", "4: Minimal framework builder",
        "精简推理框架编写工程师 Agent：写一个 PyTorch 最小正向推理框架，"
        "支持逐层按需加载 + 每层关键点 hidden_state dump。",
    ),
    PhaseMeta(
        "P5_verify_minimal", "5: Minimal framework verifier",
        "精简推理框架验证工程师 Agent：用确定输入跑推理，捕获崩溃日志，"
        "语义判断输出是否合理；失败带日志回到 P4。",
    ),
    PhaseMeta(
        "P6_port_engine", "6: Port to target framework",
        "推理引擎移植工程师 Agent：在 target_framework_dir 启动大模型，"
        "依据 P5 dump 比对相似度，替换/新增算子直到输出语义正确；每轮在"
        "target_fw_dir 下创建 git 提交。",
    ),
    PhaseMeta("finished", "finished", "run ended", is_terminal=True),
]

TRANSITIONS: Dict[Tuple[Phase, Outcome], Transition] = {
    # Forward path.
    ("P1_weight_analysis", OK): Transition(
        "P1_weight_analysis", OK, "P2_framework_analysis", label="ok"),
    ("P2_framework_analysis", OK): Transition(
        "P2_framework_analysis", OK, "P3_architect_review", label="ok"),
    ("P3_architect_review", OK): Transition(
        "P3_architect_review", OK, "P4_minimal_framework", label="ok"),
    ("P4_minimal_framework", OK): Transition(
        "P4_minimal_framework", OK, "P5_verify_minimal", label="ok"),
    ("P5_verify_minimal", OK): Transition(
        "P5_verify_minimal", OK, "P6_port_engine", label="ok"),
    ("P6_port_engine", OK): Transition(
        "P6_port_engine", OK, "finished", label="done"),

    # Architect bounce-back: P3 → P1. Pipeline caps the count.
    ("P3_architect_review", BOUNCE_BACK): Transition(
        "P3_architect_review", BOUNCE_BACK, "P1_weight_analysis",
        label="redo weight analysis"),

    # Minimal-framework repair loop: P5 fail → P4 retry (pipeline caps).
    ("P5_verify_minimal", TEST_FAIL): Transition(
        "P5_verify_minimal", TEST_FAIL, "P4_minimal_framework",
        label="repair minimal framework"),
    ("P5_verify_minimal", INFRA_FAIL): Transition(
        "P5_verify_minimal", INFRA_FAIL, "P4_minimal_framework",
        label="repair minimal framework"),

    # P6 self-repair: stays in P6 until the similarity threshold passes
    # or the pipeline cap fires.
    ("P6_port_engine", NEEDS_REPAIR): Transition(
        "P6_port_engine", NEEDS_REPAIR, "P6_port_engine",
        label="re-port iteration"),
    ("P6_port_engine", TEST_FAIL): Transition(
        "P6_port_engine", TEST_FAIL, "P6_port_engine",
        label="re-port iteration"),
    ("P6_port_engine", INFRA_FAIL): Transition(
        "P6_port_engine", INFRA_FAIL, "P6_port_engine",
        label="re-port iteration"),

    # Infra-retry on analysis phases (same phase).
    ("P1_weight_analysis", INFRA_FAIL): Transition(
        "P1_weight_analysis", INFRA_FAIL, "P1_weight_analysis", label="retry"),
    ("P2_framework_analysis", INFRA_FAIL): Transition(
        "P2_framework_analysis", INFRA_FAIL, "P2_framework_analysis",
        label="retry"),
    ("P3_architect_review", INFRA_FAIL): Transition(
        "P3_architect_review", INFRA_FAIL, "P3_architect_review",
        label="retry"),
    ("P4_minimal_framework", INFRA_FAIL): Transition(
        "P4_minimal_framework", INFRA_FAIL, "P4_minimal_framework",
        label="retry"),

    # Logic failures → terminal.
    ("P1_weight_analysis", LOGIC_FAIL): Transition(
        "P1_weight_analysis", LOGIC_FAIL, "finished", label="fail"),
    ("P2_framework_analysis", LOGIC_FAIL): Transition(
        "P2_framework_analysis", LOGIC_FAIL, "finished", label="fail"),
    ("P3_architect_review", LOGIC_FAIL): Transition(
        "P3_architect_review", LOGIC_FAIL, "finished", label="fail"),
    ("P4_minimal_framework", LOGIC_FAIL): Transition(
        "P4_minimal_framework", LOGIC_FAIL, "finished", label="fail"),
    ("P5_verify_minimal", LOGIC_FAIL): Transition(
        "P5_verify_minimal", LOGIC_FAIL, "finished", label="fail"),
    ("P6_port_engine", LOGIC_FAIL): Transition(
        "P6_port_engine", LOGIC_FAIL, "finished", label="fail"),
}


def next_transition(from_phase: Phase, outcome: Outcome) -> Optional[Transition]:
    return TRANSITIONS.get((from_phase, outcome))


def phase_label(p: Phase) -> str:
    for m in PHASES:
        if m.id == p:
            return m.label
    return str(p)


def phase_meta(p: Phase) -> Optional[PhaseMeta]:
    for m in PHASES:
        if m.id == p:
            return m
    return None


def is_terminal(p: Phase) -> bool:
    m = phase_meta(p)
    return bool(m and m.is_terminal)


def nodes_for_graph() -> List[Dict[str, str]]:
    return [
        {"id": m.id, "label": m.label, "description": m.description}
        for m in PHASES if m.id not in ("idle", "finished")
    ]


def edges_for_graph() -> List[Dict[str, str]]:
    merged: Dict[Tuple[Phase, Phase], List[str]] = {}
    for (frm, _outc), t in TRANSITIONS.items():
        merged.setdefault((frm, t.to_phase), []).append(t.label or _outc)
    out: List[Dict[str, str]] = []
    for (frm, to), labels in merged.items():
        out.append({
            "from": frm,
            "to": to,
            "label": " / ".join(sorted(set(labels))),
        })
    return out


def outcome_label(o: Outcome) -> str:
    return {
        OK: "ok",
        LOGIC_FAIL: "logic fail",
        INFRA_FAIL: "infra fail",
        TEST_FAIL: "test fail",
        BOUNCE_BACK: "bounce back",
        NEEDS_REPAIR: "needs repair",
        ABORTED: "aborted",
    }.get(o, str(o))


def graph_payload(current, last_outcome, last_label) -> Dict[str, Any]:
    nodes = nodes_for_graph()
    edges = edges_for_graph()
    active_edge = None
    if last_label:
        for e in edges:
            if e["to"] == current and last_label in e["label"].split(" / "):
                active_edge = {"from": e["from"], "to": e["to"], "label": last_label}
                break
    terminal_nodes = [
        {"id": m.id, "label": m.label, "description": m.description}
        for m in PHASES if m.is_terminal
    ]
    outcome_legend = [{"id": o, "label": outcome_label(o)} for o in ALL_OUTCOMES]
    return {
        "current": current,
        "nodes": nodes,
        "edges": edges,
        "active_edge": active_edge,
        "last_outcome": last_outcome,
        "terminal_nodes": terminal_nodes,
        "outcome_legend": outcome_legend,
    }
