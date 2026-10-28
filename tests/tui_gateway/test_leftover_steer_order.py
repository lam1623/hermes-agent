"""A steer the turn did not use runs before a prompt the user sent after it.

The turn finalizer hands an accepted steer the agent never read on as ``pending_steer``, and
``_run_post_turn_followups`` (``tui_gateway/prompt_turn.py``) requeues it through ``_enqueue_prompt``,
which appends after whatever is queued. So a prompt queued after the steer ran first: the user sent
"late" then "B", and the next turn got "B\\n\\nlate". A prompt queued before the steer stays ahead of it.

The turn runs through the real path (``_run_prompt_submit``, inline threads) with an ``AIAgent`` that
has its real steer buffer and ends its turn like the finalizer (``agent/turn_finalizer.py``: the
leftover steers become ``pending_steer``). ``during_turn`` runs while the agent still works, before
the finalizer's drain.
"""

from __future__ import annotations

import threading

import pytest

from run_agent import AIAgent
from tui_gateway import server


class _InlineThread:
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


def _agent(during_turn) -> AIAgent:
    """An ``AIAgent`` with only its steer and interrupt state. Its first turn runs ``during_turn`` (the
    user's messages while it works) and then ends without reading its steer buffer, as a turn whose last
    model call had already returned does: the finalizer hands the leftover on as ``pending_steer``."""
    agent = object.__new__(AIAgent)
    vars(agent).update(
        _pending_steer=None, _pending_steer_lock=threading.Lock(), _pending_redirect=None,
        _pending_redirect_lock=threading.Lock(), _model_request_active=threading.Event(),
        _executing_tools=False, _execution_thread_id=None, _interrupt_thread_signal_pending=False,
        _interrupt_requested=False, _interrupt_message=None, _active_children=[],
        _active_children_lock=threading.Lock(), _tool_worker_threads=None, _tool_worker_threads_lock=None,
        _supports_active_turn_redirect=True, session_id="session-key", quiet_mode=True, turns=[])

    def run_conversation(message, **_kwargs):
        agent.turns.append(message)
        if len(agent.turns) == 1:
            during_turn()
        return {"final_response": f"answered {message}", "pending_steer": agent._drain_leftover_steer()}

    agent.run_conversation = run_conversation
    agent.interrupt = lambda *args, **kwargs: None
    return agent


def _session(agent) -> dict:
    return {
        "agent": agent, "session_key": "session-key", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": True, "transport": None, "attached_images": [],
        "image_counter": 0, "cols": 80, "slash_worker": None, "show_reasoning": False,
        "tool_progress_mode": "all", "inflight_turn": None,
    }


def _run_first_turn(monkeypatch, session) -> None:
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: None)
    server._run_prompt_submit("rid", "sid", session, "first")


def _submit(session, text) -> str:
    """``prompt.submit`` on a busy session: its reply status."""
    return server._handle_busy_submit("r2", "sid", session, text, None)["result"]["status"]


def _session_steer(session, text) -> str:
    """The ``session.steer`` RPC (the client's explicit steer): its reply status."""
    server._sessions["sid"] = session
    try:
        request = {"id": "2", "method": "session.steer", "params": {"session_id": "sid", "text": text}}
        return server.handle_request(request)["result"]["status"]
    finally:
        server._sessions.pop("sid", None)


def test_a_prompt_queued_after_an_explicit_steer_runs_after_it(turn_env, monkeypatch):
    """``queue`` mode: the user steers "late" (``session.steer``) and then sends "B", which is queued
    while the turn still works."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    replies = []
    agent = _agent(during_turn=lambda: replies.extend([_session_steer(session, "late"), _submit(session, "B")]))
    session = _session(agent)

    _run_first_turn(monkeypatch, session)

    assert replies == ["queued", "queued"]
    assert agent.turns == ["first", "late\n\nB"]  # main: ["first", "B\n\nlate"]


def test_a_prompt_queued_before_the_steer_stays_ahead_of_it(turn_env, monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    replies = []
    agent = _agent(during_turn=lambda: replies.extend([_submit(session, "A"), _session_steer(session, "late")]))
    session = _session(agent)

    _run_first_turn(monkeypatch, session)

    assert replies == ["queued", "queued"]
    assert agent.turns == ["first", "A\n\nlate"]


def test_the_steer_goes_between_prompts_queued_before_and_after_it(turn_env, monkeypatch):
    """"A", then the steer "late", then "B": the next turn reads them in that order."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    replies = []
    agent = _agent(during_turn=lambda: replies.extend(
        [_submit(session, "A"), _session_steer(session, "late"), _submit(session, "B")]))
    session = _session(agent)

    _run_first_turn(monkeypatch, session)

    assert replies == ["queued", "queued", "queued"]
    assert agent.turns == ["first", "A\n\nlate\n\nB"]  # main: ["first", "A\n\nB\n\nlate"]


# The post-turn step alone: steers through ``session.steer``, prompts through the queue, then
# ``_run_post_turn_followups`` with the leftover as the turn's ``pending_steer`` (the drain held, so
# the queue can be read).


class _Steerable:
    def steer(self, text):
        return True


def _bare_session() -> dict:
    return {"agent": _Steerable(), "running": True, "history_lock": threading.Lock(), "transport": None,
            "inflight_turn": None}


def _steer(session, text) -> None:
    assert _session_steer(session, text) == "queued"


def _queue_image(session, text) -> None:
    server._enqueue_prompt(session, text, None, image_paths=[f"/nonexistent/{text}.png"])


def _after_turn(monkeypatch, session, leftover) -> list:
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda rid, sid, s: True)
    server._run_post_turn_followups("rid", "sid", session, {"pending_steer": leftover}, None)
    entries = ([session["queued_prompt"]] if session.get("queued_prompt") else []) + list(session.get("queued_prompts") or [])
    return [(e["text"], bool(e.get("image_paths"))) for e in entries]


def test_an_image_prompt_sent_after_the_steer_runs_after_it(monkeypatch):
    session = _bare_session()
    _steer(session, "late")
    _queue_image(session, "B")

    assert _after_turn(monkeypatch, session, "late") == [("late", False), ("B", True)]  # main: B first


def test_a_steer_after_an_image_prompt_goes_between_it_and_what_followed(monkeypatch):
    session = _bare_session()
    _queue_image(session, "A")
    _steer(session, "late")
    server._enqueue_prompt(session, "C", None)

    assert _after_turn(monkeypatch, session, "late") == [("A", True), ("late", False), ("C", False)]  # main: A, C, late


def test_each_unread_steer_goes_back_to_its_own_place(monkeypatch):
    session = _bare_session()
    _steer(session, "s1")
    server._enqueue_prompt(session, "A", None)
    _steer(session, "s2")
    server._enqueue_prompt(session, "B", None)

    assert _after_turn(monkeypatch, session, "s1\ns2") == [("s1\n\nA\n\ns2\n\nB", False)]  # main: A, B, s1, s2


def test_a_steer_the_agent_already_read_is_not_placed_again(monkeypatch):
    """The agent read "s1" at a tool boundary; only "s2" is left over."""
    session = _bare_session()
    _steer(session, "s1")
    server._enqueue_prompt(session, "A", None)
    _steer(session, "s2")
    server._enqueue_prompt(session, "B", None)

    assert _after_turn(monkeypatch, session, "s2") == [("A\n\ns2\n\nB", False)]  # main: A, B, s2


def test_a_leftover_no_steer_here_accounts_for_is_queued_last_as_before(monkeypatch):
    """A redirect the agent turned into a steer did not pass through ``_steer_live_turn``."""
    session = _bare_session()
    server._enqueue_prompt(session, "A", None)
    _steer(session, "late")

    assert _after_turn(monkeypatch, session, "redirected") == [("A\n\nredirected", False)]
