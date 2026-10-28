"""A prompt sent between a turn's ``message.complete`` and its busy release runs as the next turn.

``tui_gateway/prompt_turn.py`` emits ``message.complete`` before it releases ``running``, so a client that
sends its next prompt on that event reaches the busy path while the session still reads busy. In ``steer``
mode the prompt was steered into a turn whose agent had already drained its leftover steers, and nothing
ran it; ``interrupt`` mode queued it but sent a hard interrupt to that finished turn. These tests hold the
turn at that point: the ``message.complete`` emit submits the next prompt, as the stdio client does,
before the turn releases. The agent is an ``AIAgent`` with its real steer buffer
(``agent/interrupt_control.py``) whose turn ends the way the finalizer does (``agent/turn_finalizer.py``:
the leftover steers become ``pending_steer``).
"""

from __future__ import annotations

import threading

import pytest

from run_agent import AIAgent
from tui_gateway import server


class _InlineThread:
    """Turn and interrupt workers run inline, so the test reads the state they leave."""

    def __init__(self, target=None, args=(), kwargs=None, **_ignored):
        self._call = (target, args, kwargs or {})

    def start(self):
        target, args, kwargs = self._call
        if target is not None:
            target(*args, **kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    """The turn pipeline without its environment-heavy side paths (as in ``test_auto_continue.py``)."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    stubs = {
        "_wire_callbacks": lambda sid: None,
        "_sync_agent_model_with_config": lambda sid, session: None,
        "_session_cwd": lambda session: str(tmp_path),
        "_register_session_cwd": lambda session: None,
        "_tts_stream_begin": lambda: None,
        "_sync_session_key_after_compress": lambda *a, **k: None,
        "_get_usage": lambda agent: {},
        "_session_compression_in_flight": lambda session: False,
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(server, name, stub)


def _agent(in_turn_tail=None) -> AIAgent:
    """An ``AIAgent`` with only its steer and interrupt state (the ``tests/agent/test_steer.py`` stub). Its
    turn records the prompt, drains the leftover steers like the finalizer, then runs ``in_turn_tail``
    (the finalizer's later steps: memory sync, background review, hooks)."""
    agent = object.__new__(AIAgent)
    vars(agent).update(
        _pending_steer=None, _pending_steer_lock=threading.Lock(), _pending_redirect=None,
        _pending_redirect_lock=threading.Lock(), _model_request_active=threading.Event(),
        _executing_tools=False, _execution_thread_id=None, _interrupt_thread_signal_pending=False,
        _interrupt_requested=False, _interrupt_message=None, _active_children=[],
        _active_children_lock=threading.Lock(), _tool_worker_threads=None, _tool_worker_threads_lock=None,
        _supports_active_turn_redirect=True, session_id="session-key", quiet_mode=True,
        turns=[], interrupts=[])

    def run_conversation(message, **_kwargs):
        agent.turns.append(message)
        leftover = agent._drain_leftover_steer()
        if in_turn_tail is not None:
            in_turn_tail()
        return {"final_response": f"answered {message}", "pending_steer": leftover}

    agent.run_conversation = run_conversation
    agent.interrupt = lambda *args, **kwargs: agent.interrupts.append(args)
    return agent


def _session(agent) -> dict:
    # running=True: prompt.submit claims the session before it starts the turn.
    return {
        "agent": agent, "session_key": "session-key", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": True, "transport": None, "attached_images": [],
        "image_counter": 0, "cols": 80, "slash_worker": None, "show_reasoning": False,
        "tool_progress_mode": "all", "inflight_turn": None,
    }


def _run_first_turn(monkeypatch, session, on_complete=None) -> None:
    """Run the prompt "first" through the real turn path. ``on_complete`` runs inside the first
    ``message.complete`` emit, while the turn has not released ``running`` yet."""
    completes = []

    def emit(event, sid, payload=None):
        if event != "message.complete":
            return
        completes.append(payload)
        if on_complete is not None and len(completes) == 1:
            assert session["running"], "the turn is held between message.complete and its release"
            on_complete()

    monkeypatch.setattr(server, "_emit", emit)
    server._run_prompt_submit("rid", "sid", session, "first")


def _submit(session, text) -> str:
    """``prompt.submit`` on a busy session: its reply status."""
    return server._handle_busy_submit("r2", "sid", session, text, None)["result"]["status"]


@pytest.mark.parametrize("mode", ["steer", "interrupt"])
def test_a_prompt_sent_on_message_complete_runs_as_the_next_turn(turn_env, monkeypatch, mode):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: mode)
    agent = _agent()
    session = _session(agent)
    replies = []

    _run_first_turn(monkeypatch, session, on_complete=lambda: replies.append(_submit(session, "second")))

    # Main, steer mode: "steered" into the ended turn, and only "first" ever ran.
    assert replies == ["queued"]
    assert agent.turns == ["first", "second"]
    # Main, interrupt mode: the prompt ran, after a hard interrupt sent to the finished turn.
    assert agent.interrupts == []
    assert agent._pending_steer is None
    assert not session["running"]


def test_a_steer_accepted_after_the_agent_drained_its_leftovers_runs_as_the_next_turn(turn_env, monkeypatch):
    """Before ``message.complete``: the finalizer has drained the steer buffer, the gateway still runs the
    turn, and a steer lands in that buffer. It must not wait there for a turn that never reads it."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    replies = []

    def finalizer_tail():
        if len(agent.turns) == 1:
            replies.append(_submit(session, "late"))

    agent = _agent(in_turn_tail=finalizer_tail)
    session = _session(agent)

    _run_first_turn(monkeypatch, session)

    assert replies == ["steered"]
    assert agent.turns == ["first", "late"]  # main: ["first"], "late" left in the steer buffer
    assert agent._pending_steer is None


def test_the_same_words_sent_again_on_message_complete_are_a_new_prompt(turn_env, monkeypatch):
    """The finished turn's prompt is no longer live: the same text sent again (a second "continue") is not
    a client's copy of a running prompt to drop."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = _agent()
    session = _session(agent)
    replies = []

    _run_first_turn(monkeypatch, session, on_complete=lambda: replies.append(_submit(session, "first")))

    assert replies == ["queued"]
    assert agent.turns == ["first", "first"]


@pytest.mark.parametrize(("entry", "goes_out_as"), [("session.steer", "rejected"), ("/steer", "send")])
def test_an_explicit_steer_on_message_complete_goes_out_as_the_next_prompt(turn_env, monkeypatch, entry,
                                                                           goes_out_as):
    """``session.steer`` and ``/steer`` answer as for an idle session ("rejected", "send": the client sends
    the text as its next prompt) instead of parking it in the finished turn's steer buffer."""
    agent = _agent()
    session = _session(agent)
    replies = []

    def steer():
        server._sessions["sid"] = session
        try:
            if entry == "session.steer":
                request = {"id": "2", "method": "session.steer", "params": {"session_id": "sid", "text": "second"}}
                replies.append(server.handle_request(request)["result"]["status"])
            else:
                params = {"name": "steer", "arg": "second", "session_id": "sid"}
                replies.append(server._methods["command.dispatch"]("2", params)["result"]["type"])
        finally:
            server._sessions.pop("sid", None)

    _run_first_turn(monkeypatch, session, on_complete=steer)

    assert replies == [goes_out_as]  # main: "queued" / "exec", the text parked in the buffer
    assert agent._pending_steer is None
    assert agent.turns == ["first"]
