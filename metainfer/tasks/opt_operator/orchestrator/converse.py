"""Create-time conversational requirement confirmation (R1).

The New-run flow for ``opt-operator`` is a chat, not a flat form. The user
describes the operator they want optimized in free text; the task "listens",
extracts as much structure as it can, flags what is still missing or ambiguous,
and keeps asking until everything required is pinned down. Only then does it
present an editable *interpretation card* and, on confirmation, emit the exact
flat ``answers`` + ``raw_request`` transcript that ``POST /tasks`` stores.

This module is the pure, deterministic heart of that conversation — it is
deliberately **not** an LLM. It reads a normalized form *schema* (the same
``load_form_schema`` JSON the shell renders) and:

  * detects categorical values in free text by matching each select/radio
    field's declared option labels, augmented by a small operator lexicon
    (``hip``/``cuda`` → ``HIP``, ``targeted``/``specialize`` → targeted, …);
  * never guesses on a field whose text names two conflicting options;
  * tracks which *required* fields are still unsatisfied and turns each into a
    concrete follow-up question;
  * settles into a flat answers dict that satisfies the shell's form
    validator (required present, ``select`` values within the declared option
    labels), plus the dialogue transcript for ``requirements.json::raw_request``.

No filesystem, no LLM, no GPU — unit-testable in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# -- conversation API ------------------------------------------------------ #
#
# ``schema`` is the normalized form schema:
#   {"type":..., "fields": [ {key,label,help,type,required,default,options,override_component} ]}
# A categorical field's valid answer is one of its ``options[].label`` strings
# (matches the shell validator exactly).

# Alias lexicon: field.key -> { cue -> canonical option label }. Cues are
# matched as whole words (case-insensitive) inside the user's message.
_LEXICON: Dict[str, Dict[str, str]] = {
    "kernel_language": {
        "hip": "HIP", "cuda": "HIP", "c++": "HIP", "gfx928": "HIP",
        "dcu": "HIP", "amd": "HIP", "triton": "Triton",
    },
    "shape_mode": {
        "targeted": "Targeted (specific shape)",
        "specific": "Targeted (specific shape)",
        "specialize": "Targeted (specific shape)",
        "specialised": "Targeted (specific shape)",
        "single shape": "Targeted (specific shape)",
        "one shape": "Targeted (specific shape)",
        "fixed shape": "Targeted (specific shape)",
        "general": "General (any shape)",
        "generic": "General (any shape)",
        "any shape": "General (any shape)",
        "sweep": "General (any shape)",
        "all shapes": "General (any shape)",
    },
    "input_mode": {
        "i have a kernel": "I have a kernel source",
        "existing kernel": "I have a kernel source",
        "kernel source": "I have a kernel source",
        "source code": "I have a kernel source",
        "my kernel": "I have a kernel source",
        "from scratch": "Spec only (from scratch)",
        "no kernel": "Spec only (from scratch)",
        "just a spec": "Spec only (from scratch)",
        "spec only": "Spec only (from scratch)",
    },
}

# A lexicon of fields whose value is a count we pull straight out of prose.
_NUMBER_FIELDS = {"num_gpus": r"\b([0-9]+)\s*(?:gpu|gpus|dcu|dcus)\b",
                  "max_iterations": r"\b([0-9]+)\s*(?:iteration|iterations|round|rounds|cycle|cycles)\b"}


@dataclass(frozen=True)
class CardItem:
    """One editable row of the interpretation card."""
    key: str
    label: str
    value: Any
    kind: str                       # text|textarea|select|multiselect|file|number
    required: bool
    options: Optional[List[str]]
    origin: str                     # user | default | pending
    note: str = ""


@dataclass(frozen=True)
class Interpretation:
    """The full state of one conversation step."""
    answers: Dict[str, Any]
    card: List[CardItem]
    missing: List[str]              # required keys still unsatisfied
    conflict: List[str]             # required keys with a conflicting reading
    open_questions: List[str]       # follow-up questions to drive the chat
    complete: bool                  # every required key pinned down
    assistant: str                  # assistant line to append to the chat
    transcript: List[Dict[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Field helpers
# --------------------------------------------------------------------------- #

def _opt_labels(field: Dict[str, Any]) -> List[str]:
    opts = field.get("options") or []
    return [str(o.get("label") if isinstance(o, dict) else o) for o in opts]


def _is_categorical(field: Dict[str, Any]) -> bool:
    return field.get("type") in ("select", "radio") and bool(field.get("options"))


def _words(text: str) -> List[str]:
    return [w for w in re.split(r"[^A-Za-z0-9+]+", text.lower()) if w]


def _match_option_label(label: str, tokens: List[str]) -> bool:
    """True if every word of the option label appears as a token (allows the
    user to say 'hip' instead of the full 'HIP' label, or 'general' for
    'General (any shape)')."""
    lwords = [w for w in re.split(r"[^A-Za-z0-9]+", label.lower()) if w]
    return all(w in tokens for w in lwords)


def _lexicon_hits(field: Dict[str, Any], tokens: List[str]) -> List[str]:
    """Canonical option labels a user's words alias to for this field."""
    hits: List[str] = []
    cues = _LEXICON.get(field.get("key") or "", {})
    for cue, label in cues.items():
        cwords = _words(cue)
        if all(cw in tokens for cw in cwords):
            hits.append(label)
    return hits


def _literal_hits(field: Dict[str, Any], tokens: List[str]) -> List[str]:
    """Option labels whose own words all appear in the message."""
    hits: List[str] = []
    for label in _opt_labels(field):
        if _match_option_label(label, tokens):
            hits.append(label)
    return hits


def _detect_categorical(field: Dict[str, Any], text: str) -> Optional[str]:
    """Return the option label to fill, or None when the text gives no / a
    conflicting answer for this field."""
    tokens = _words(text)
    hits = _lexicon_hits(field, tokens) + _literal_hits(field, tokens)
    if not hits:
        return None
    # A conflict (both HIP and Triton mentioned) must not be auto-resolved.
    uniq = list(dict.fromkeys(hits))
    return uniq[0] if len(uniq) == 1 else None


def _detect_number(field: Dict[str, Any], text: str) -> Optional[str]:
    pat = _NUMBER_FIELDS.get(field.get("key") or "")
    if not pat:
        return None
    m = re.search(pat, text, re.IGNORECASE)
    return str(int(m.group(1))) if m else None


_CONTRACT_KEYS = ("name", "entrypoint", "inputs", "outputs", "shapes", "forward", "op")


def _detect_contract(text: str) -> Optional[str]:
    """Extract an ``operator_contract`` value from free text.

    We only accept a contract when the user actually pastes a contract-like
    YAML block (a line starting with a top-level key such as name/inputs/
    shapes). We never fabricate a contract from bare prose. When present we
    isolate the YAML block itself (preferring a ``` fenced block, else slicing
    from the first top-level key line) so surrounding chat prose does NOT leak
    into the value — the contract later feeds ``OperatorContract.load``.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    # A fenced block wins when present: ```yaml ... ``` / ``` ... ```
    fences = re.findall(r"```[A-Za-z]*[ \t]*\n(.*?)```", text, re.DOTALL)
    if fences:
        return fences[0].strip()
    lines = text.splitlines()
    keypat = "|".join(_CONTRACT_KEYS)
    start = next((i for i, ln in enumerate(lines)
                  if re.match(rf"^\s*({keypat})\s*:", ln)), None)
    if start is None:
        return None
    # Take the block from the first top-level key line onward, but STOP before
    # trailing chat prose: after a blank line, a non-indented line that is not a
    # YAML key ends the contract (e.g. "thanks!"). Indented lines are block
    # continuations and are kept.
    kept: List[str] = []
    for ln in lines[start:]:
        indented = bool(ln[:1] in (" ", "\t"))
        key_line = bool(re.match(rf"({keypat})\s*:", ln))
        if not indented and ln.strip() and not key_line:
            break  # trailing prose at column 0
        kept.append(ln)
    return "\n".join(kept).strip()


# --------------------------------------------------------------------------- #
# Interpretation / settle
# --------------------------------------------------------------------------- #

def _valid_default(field: Dict[str, Any], value: Any) -> bool:
    """A schema default only seeds an answer when it is a *valid* value for the
    field. Select/radio answers must equal an option label; form.yaml string
    defaults like ``"spec"`` / ``"triton"`` are display hints, not labels, so
    they must NOT be emitted (the shell validator would reject them)."""
    if value is None or value == "":
        return False
    if _is_categorical(field):
        return value in _opt_labels(field)
    return True


def _starter_answers(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Seed answers from schema defaults that are actually valid for the field."""
    out: Dict[str, Any] = {}
    for f in schema.get("fields") or []:
        d = f.get("default")
        if _valid_default(f, d):
            out[f["key"]] = d
    return out


def interpret(
    schema: Dict[str, Any],
    *,
    request: str = "",
    answers: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    transcript: Optional[List[Dict[str, str]]] = None,
) -> Interpretation:
    """Advance the conversation by one user message.

    ``answers`` is the card state carried between turns; new slot extractions
    from ``request`` are merged over it. ``defaults`` may pre-supply values
    (e.g. a previous conversation's confirmed answers). ``transcript`` is the
    running dialogue (each ``{"role", "text"}``), retained so ``settle`` can
    emit ``requirements.json::raw_request``. Returns an
    :class:`Interpretation` the wizard renders.
    """
    ans: Dict[str, Any] = _starter_answers(schema)
    if defaults:
        for k, v in defaults.items():
            if v is not None and v != "":
                ans[k] = v
    if answers:
        for k, v in answers.items():
            if v is not None and v != "":
                ans[k] = v
    if request:
        ans = _extract(schema, ans, request)

    card, missing, conflict = _render_card(schema, ans)
    complete = not missing and not conflict

    open_q = _follow_ups(schema, missing, conflict, card)
    if not request:
        assistant = _opening(schema)
    elif not complete and open_q:
        assistant = open_q[0]
    elif conflict:
        assistant = conflict[0]
    else:
        assistant = ("Everything I need is pinned down. Review the summary "
                     "below, edit anything that is wrong, then confirm to start.")
    return Interpretation(
        answers=dict(ans), card=card, missing=missing, conflict=conflict,
        open_questions=open_q, complete=complete, assistant=assistant,
        transcript=list(transcript or []),
    )


def _extract(schema: Dict[str, Any], ans: Dict[str, Any],
             request: str) -> Dict[str, Any]:
    """Pull every reliably-detectable slot out of one free-text message."""
    out = dict(ans)
    for f in schema.get("fields") or []:
        key = f["key"]
        if _is_categorical(f):
            val = _detect_categorical(f, request)
            if val is not None:
                out[key] = val
        elif f.get("type") in ("number",):
            val = _detect_number(f, request)
            if val is not None:
                out[key] = val
        elif key == "operator_contract":
            val = _detect_contract(request)
            if val is not None:
                out[key] = val
        # file / textarea / text fields without reliable prose parsing are left
        # for the interpretation card unless matched above.
    return out


def _render_card(schema: Dict[str, Any], ans: Dict[str, Any]):
    card: List[CardItem] = []
    missing: List[str] = []
    conflict: List[str] = []
    for f in schema.get("fields") or []:
        key = f["key"]
        val = ans.get(key)
        opts = _opt_labels(f)
        required = bool(f.get("required"))
        kind = f.get("type") or "text"
        empty = val is None or val == "" or (isinstance(val, str) and not val.strip())
        # File pickers can't be satisfied by typing; mark the upload path.
        if kind == "file":
            if required and empty:
                missing.append(key)
            card.append(CardItem(
                key=key, label=f.get("label") or key, value=val, kind=kind,
                required=required, options=opts or None,
                origin="default" if empty else "user",
                note="attach the source/reference file" if empty else ""))
            continue
        if required and empty:
            missing.append(key)
            card.append(CardItem(key=key, label=f.get("label") or key,
                                 value=val, kind=kind, required=True,
                                 options=opts or None, origin="pending",
                                 note="required — still needed"))
            continue
        if required and kind in ("select", "radio") and opts and val not in opts:
            conflict.append(key)
            card.append(CardItem(key=key, label=f.get("label") or key,
                                 value=val, kind=kind, required=True,
                                 options=opts or None, origin="user",
                                 note=f"must be one of {opts}"))
            continue
        card.append(CardItem(key=key, label=f.get("label") or key, value=val,
                             kind=kind, required=required, options=opts or None,
                             origin="user" if not empty else "default"))
    return card, missing, conflict


def _opening(schema: Dict[str, Any]) -> str:
    name = (schema.get("label") or schema.get("type") or "this task")
    return (f"Hi — I optimise kernels for the K100 (HIP/Triton). "
            f"Describe the operator you want to optimise: paste a contract "
            f"(name/inputs/outputs/shapes YAML), give me your existing kernel "
            f"source, or say which operator + shapes and stack you have in mind.")


def _follow_ups(schema: Dict[str, Any], missing: List[str], conflict: List[str],
                card: List[CardItem]) -> List[str]:
    by_key = {c.key: c for c in card}
    out: List[str] = []
    for key in conflict:
        c = by_key.get(key)
        out.append((f"'{key}' currently reads {c.value!r}, which isn't a valid "
                    f"choice. Pick one of: {', '.join(c.options or [])}."))
    for key in missing:
        c = by_key.get(key)
        if key == "operator_contract":
            out.append("I still need the operator contract. Paste YAML with "
                       "name/entrypoint/inputs/outputs/shapes (or drop your "
                       "kernel source file and I'll derive its signature).")
        elif key == "kernel_source":
            out.append("Upload the kernel source file to optimise.")
        else:
            label = c.label if c else key
            out.append(f"Please tell me: {label}. " +
                       (f"One of: {', '.join(c.options or [])}." if c and c.options else ""))
    # Keep the turn focused: surface the most important open question first.
    return out


def settle(schema: Dict[str, Any], answers: Dict[str, Any],
           transcript: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Final step after the user confirms the card.

    Returns the payload ``POST /tasks`` expects: flat ``answers`` plus a
    ``raw_request`` dialogue transcript. Raises :class:`ValueError` if any
    required field is still empty so a buggy wizard can never create a task the
    shell would reject.
    """
    interp = interpret(schema, request="", answers=answers)
    if interp.missing:
        raise ValueError(
            f"cannot settle: still missing required fields: {interp.missing}")
    if interp.conflict:
        raise ValueError(
            f"cannot settle: invalid values to fix on the card: {interp.conflict}")
    tr = list(transcript or [])
    raw_request = _format_transcript(tr)
    return {"answers": interp.answers, "raw_request": raw_request}


def _format_transcript(turns: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for t in turns:
        role = (t.get("role") or "?")
        text = (t.get("text") or "").strip()
        if text:
            lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def _satisfied_fields(schema: Dict[str, Any]) -> List[str]:
    """Required keys whose schema option list a value must land in."""
    return [f["key"] for f in schema.get("fields") or []
            if _is_categorical(f) and f.get("required")]


__all__ = ["Interpretation", "CardItem", "interpret", "settle",
           "_satisfied_fields"]
