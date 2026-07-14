"""In-process mocks for the orchestrator's agent + state interfaces.

These mocks let step code (step0_rough / step3_calculate / pipeline)
fan out via ``launch_async`` without ever spawning a subprocess —
``launch_async`` resolves the canned response inline and invokes the
``on_done`` callback in a daemon thread.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from metainfer.orchestrator.subagent_manager import AgentResult


class MockAgentManager:
    """In-process stand-in for ``SubAgentManager``.

    Construction signature mirrors SubAgentManager only loosely — tests
    typically pass no args. Behaviour:

    * ``launch_async(spec, on_done=None)``: looks up the canned response
      for ``spec.name`` in ``self.responses`` (or invokes
      ``self.response_fn(spec)`` if set), wraps it in an ``AgentResult``,
      stores it in ``self.results`` and invokes ``on_done`` if provided.
      Returns a ``threading.Thread`` that is already "done" (the lookup
      happens inline). The thread's ``join()`` is a no-op so step code
      that calls ``t.join()`` returns immediately.
    * ``result(name)``: returns the stored ``AgentResult`` (or ``None``).
    * ``snapshot()``: returns ``[]`` — we never have live agents.
    * ``kill(name)`` / ``kill_all()`` / ``shutdown()``: no-ops.

    The mock is single-threaded by design (no actual subprocesses); step
    code that fans out via ``launch_async`` still works because each call
    immediately resolves and stores its result.
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        response_fn: Optional[Callable[[Any], str]] = None,
        failures: Optional[Dict[str, str]] = None,
        delay_s: float = 0.0,
        synthesize_session_id: bool = True,
    ) -> None:
        # name -> final_text
        self.responses: Dict[str, str] = dict(responses or {})
        # name -> error string (the agent "fails" with this message)
        self.failures: Dict[str, str] = dict(failures or {})
        self.response_fn = response_fn
        self.delay_s = delay_s
        # When True, every successful AgentResult gets a synthesized
        # session_id (so callers that thread resume_session_id through
        # turns — like AgentPool — have something to chain). Turn off
        # for tests that explicitly want None.
        self.synthesize_session_id = synthesize_session_id
        self.results: Dict[str, AgentResult] = {}
        self.launched_specs: List[Any] = []
        self._lock = threading.Lock()

    # -- helpers for tests --------------------------------------------------

    def add_response(self, name: str, text: str) -> None:
        self.responses[name] = text

    def add_failure(self, name: str, error: str) -> None:
        self.failures[name] = error

    def _resolve(self, spec: Any) -> AgentResult:
        if spec.name in self.failures:
            return AgentResult(
                name=spec.name, role=spec.role, success=False,
                returncode=1, duration_s=0.01,
                error=self.failures[spec.name],
                attempts=1, failure_mode="logic",
            )
        if spec.name in self.responses:
            text = self.responses[spec.name]
        elif self.response_fn is not None:
            text = self.response_fn(spec)
        else:
            text = ""
        # Echo back the resume_session_id so the pool (or any caller
        # that chains turns via resume_session_id) sees a stable
        # session id — mimicking what real ccb does on --resume.
        # For turn-0 specs (no resume), mint a synthetic id.
        sid = None
        if self.synthesize_session_id:
            sid = spec.resume_session_id or f"mock-sess-{spec.name}"
        return AgentResult(
            name=spec.name, role=spec.role, success=True,
            returncode=0, duration_s=0.01,
            final_text=text, attempts=1, session_id=sid,
        )

    # -- SubAgentManager-shaped API ----------------------------------------

    def launch(self, spec: Any):
        """Blocking variant of ``launch_async`` — used by AgentPool.

        Records the spec, resolves the canned result, stores it. The
        AgentPool's worker thread calls this serially per worker.
        """
        if self.delay_s:
            time.sleep(self.delay_s)
        with self._lock:
            self.launched_specs.append(spec)
            result = self._resolve(spec)
            self.results[spec.name] = result
        # Return a lightweight handle-like object: callers that need
        # the result fetch it via result(name). SubAgentManager.launch
        # returns an AgentHandle; we don't bother — nothing in the
        # pool code reads the return value.
        return None

    def launch_async(self, spec: Any, on_done: Optional[Callable] = None):
        if self.delay_s:
            time.sleep(self.delay_s)
        with self._lock:
            self.launched_specs.append(spec)
            result = self._resolve(spec)
            self.results[spec.name] = result

        def _run() -> None:
            if on_done is not None:
                on_done(result)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def result(self, name: str) -> Optional[AgentResult]:
        return self.results.get(name)

    def results(self) -> Dict[str, AgentResult]:
        return dict(self.results)

    def snapshot(self) -> List[Dict[str, Any]]:
        return []

    def kill(self, name: str) -> bool:  # noqa: D401
        return False

    def kill_all(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def health(self, name: str) -> str:
        if name in self.results:
            return "done" if self.results[name].success else "failed"
        return "unknown"


class FakeStore:
    """Stand-in for the orchestrator's StateStore.

    The step modules call ``store.append_timeline(event_type, payload)``.
    We just collect those events for assertions.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def append_timeline(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "ts": time.time(),
            "type": event_type,
            "payload": payload or {},
        })

    def types(self) -> List[str]:
        return [e["type"] for e in self.events]
