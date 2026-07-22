"""NFS-safe filesystem primitives for cross-host coordination.

Design decisions (see ``docs/multi-node-architecture.md`` and the plan file):

1. **Atomic claim = ``os.link``** (decision 1). tmp file + ``os.link(tmp, target)``;
   NFS server-side link is atomic, exactly one producer wins (others EEXIST).
   **Never use fcntl.flock across hosts** — it is unreliable on NFS.

2. **Atomic write = tmp + ``os.replace``** (decision 1 helper). tmp filename is
   uuid-suffixed to avoid same-process threaded races on the same target path.

3. **Lease token = claim file content** (decision 3). Claim files carry
   ``{holder, job_id, acquired_at, lease_until, secret}``. ``release_gpus`` must
   present the matching secret; ``force_release`` and ``reap_expired_claims`` bypass.

4. **Heartbeat = mtime only** (decision 8). The heartbeat file's mtime is the
   liveness signal; its content is never rewritten. Workers touch it every N seconds;
   readers compare mtime to now.

These primitives are pure utilities and do NOT depend on the path/layout module
(:mod:`metainfer.cluster.paths`) — callers pass absolute paths.
"""

from __future__ import annotations

import json
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Atomic write (tmp + os.replace)
# --------------------------------------------------------------------------- #
def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path``.

    Uses uuid-suffixed tmp file in the same directory (required for ``os.replace``
    to be atomic on POSIX / NFS), then ``os.replace``. Concurrent writers to the
    same path do not collide because each gets its own tmp file.

    Authority-source semantics: the caller is declaring ``path`` the SSOT for
    this data. Readers either see the old content or the new content, never a
    partial write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def atomic_write_json(path: Path, obj: Any) -> None:
    """Atomically write ``obj`` as indented JSON to ``path``. See :func:`atomic_write_text`."""
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True))


# --------------------------------------------------------------------------- #
# Atomic claim via os.link
# --------------------------------------------------------------------------- #
def link_claim(target_path: Path, payload: Dict[str, Any]) -> bool:
    """Atomically claim ``target_path`` by hard-linking a tmp file carrying ``payload``.

    Returns ``True`` if this caller won the claim (i.e. created the link),
    ``False`` if another caller already holds it (EEXIST).

    Any other OSError propagates. The tmp file is always cleaned up.

    Authority-source semantics: ``target_path`` becomes the SSOT for the claim
    once this returns True. The link itself is the mutex; the file content is
    metadata (holder identity, lease, secret) that other callers will read to
    decide whether the claim is stale / reapable.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, target_path)
            return True
        except FileExistsError:
            return False
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def read_claim(target_path: Path) -> Optional[Dict[str, Any]]:
    """Read a claim file and parse its JSON content.

    Returns ``None`` if the file is missing, unreadable, or contains malformed
    JSON (corruption tolerance — partial writes should not crash readers).

    Note: ``link_claim`` writes atomically, so a well-formed claim file is never
    partially written; this defensiveness covers filesystem corruption, NFS
    hiccups, and pre-existing badly-formed files.
    """
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None


def break_claim(target_path: Path, expected_secret: Optional[str] = None) -> bool:
    """Remove a claim file.

    If ``expected_secret`` is provided, the claim's ``secret`` field must match
    it; otherwise the unlink is skipped (returns False). This is the
    ``LeaseToken`` verification path — a holder may only release its own claim.

    Without ``expected_secret`` (admin force-release / stale-claim reap), the
    unlink proceeds unconditionally.

    Idempotent: returns ``True`` if the file existed and was removed (or already
    gone), ``False`` only on secret mismatch.
    """
    target_path = Path(target_path)
    if expected_secret is not None:
        claim = read_claim(target_path)
        if claim is None:
            # File missing — treat as already released.
            return True
        if claim.get("secret") != expected_secret:
            return False
    try:
        target_path.unlink()
    except FileNotFoundError:
        pass
    return True


# --------------------------------------------------------------------------- #
# Heartbeat (mtime only)
# --------------------------------------------------------------------------- #
def touch_heartbeat(path: Path) -> None:
    """Update heartbeat file mtime to now. Create empty if missing.

    The heartbeat file is the SSOT for "this worker is alive". Workers call this
    every N seconds. Readers use :func:`is_stale_heartbeat` to decide liveness.

    Content of the heartbeat file is never rewritten — mtime is the only signal
    (avoids atomic-write contention on a hot path).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Create with exclusive create to avoid races; content is irrelevant.
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            pass
    try:
        os.utime(path, None)  # Sets atime + mtime to now.
    except OSError:
        # Best-effort — if utime fails (permission / race), the create above
        # already established a recent mtime.
        pass


def is_stale_heartbeat(path: Path, stale_after_s: float = 60.0) -> bool:
    """Return True if heartbeat file is missing or older than ``stale_after_s`` seconds.

    Used by:
    - scoreboard reaper (decide whether to break an expired claim)
    - worker registry liveness view (mark worker dead)
    """
    path = Path(path)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return True
    except OSError:
        return True
    return (time.time() - mtime) > stale_after_s


def generate_secret() -> str:
    """Generate a fresh lease secret. Used by :func:`metainfer.cluster.scoreboard.acquire_gpus`."""
    return secrets.token_hex(16)
