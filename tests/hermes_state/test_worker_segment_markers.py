"""Which creation markers a compression segment carries when a managed worker publishes it.

``_branched_from`` and ``_reset_from`` name the edge that
made a row its own conversation (/branch, /new); a compression segment is the same conversation,
so it carries neither. The owner's in-process publish (``_publish_child_session_row``) never
copied them, but the managed-worker publish (``worker_publish``) copied both from the parent row,
and every reader that counts them by presence then treated each segment as a new conversation:
the publish itself started the segment unpinned and unarchived, the reset listing showed one chat
per segment, and list payloads lifted the marker for the app. ``_delegate_from`` is different: a
delegate's segments are still its execution transcript (``delegate_tool`` seeds the marker into the
child agent's model_config, so the in-process publish carries it too), and completion ownership
walks it out of that transcript, so the worker keeps it.
"""

import json

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError
from hermes_state_worker_compression import worker_publish

_EXECUTION = "worker-execution"


@pytest.fixture()
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    store.create_session("origin", source="cli")
    store.append_message("origin", role="user", content="the chat everything came from")
    yield store
    store.close()


def _worker_compress(store, sid, child, model_config=None):
    """Rotate *sid* to *child* the way a managed worker does: it holds the compression lease and its
    ``compression.publish`` receipt runs ``worker_publish`` on the owner's write connection."""
    holder = f"lease-{sid}"
    assert store.try_acquire_compression_lock(sid, holder)
    payload = dict(child_session_id=child, source="cli", model="m", system_prompt=None, cwd=None,
                   profile_name=None, messages=[{"role": "user", "content": f"[summary of {sid}]"}],
                   model_config={"max_iterations": 90, **(model_config or {})},
                   compression_lock_holder=holder, require_compression_lease=True,
                   require_lease_refresh=False, lease_ttl_seconds=300.0, watermark=None,
                   watermark_ceiling=None)

    def write(conn):
        # The execution follows the tip: worker_publish moves it to each new segment.
        conn.execute("INSERT OR IGNORE INTO worker_executions (execution_id, session_id, kind, owner_epoch,"
                     " generation, status, adoption_digest) VALUES (?, ?, 'compute', 1, 1, 'running', 'd')",
                     (_EXECUTION, sid))
        return worker_publish(store, conn, sid, payload)

    store._execute_write(write)
    store.release_compression_lock(sid, holder)  # the worker's compression.lock.release


def _finish_worker(store):
    store._execute_write(lambda conn: conn.execute(
        "UPDATE worker_executions SET status = 'terminal' WHERE execution_id = ?", (_EXECUTION,)))


def _config(store, sid):
    return json.loads(store.get_session(sid)["model_config"] or "{}")


def _remaining(store, ids):
    return sorted(sid for sid in ids if store.get_session(sid) is not None)


def test_a_worker_compressed_branch_is_one_pinned_conversation(db):
    db.end_session("origin", "branched")
    db.create_session("b1", source="cli", parent_session_id="origin", model_config={"_branched_from": "origin"})
    db.append_message("b1", role="user", content="branch turn")
    assert db.set_session_pinned("b1", True)
    _worker_compress(db, "b1", "b2")
    _worker_compress(db, "b2", "b3")

    # The pin is lineage-wide; a segment published after it joins it.
    assert [db.get_session(sid)["pinned"] for sid in ("b1", "b2", "b3")] == [1, 1, 1]
    rows = db.list_sessions_rich(limit=50, order_by_last_active=True, compact_rows=True, include_pinned=True)
    assert sorted(row["id"] for row in rows) == ["b3", "origin"]
    assert {row["id"]: row for row in rows}["b3"]["_lineage_ids"] == ["b1", "b2", "b3"]
    assert "_branched_from" not in db.get_session("b3")  # list payloads lift only a real marker
    assert _config(db, "b1")["_branched_from"] == "origin"  # the /branch edge keeps its marker
    for segment in ("b2", "b3"):
        assert _config(db, segment) == {"max_iterations": 90}
    _finish_worker(db)
    assert db.delete_sessions(["b3"], exclude_active_write_guards=True, include_compression_chain=True) == 1
    assert _remaining(db, ("origin", "b1", "b2", "b3")) == ["origin"]


def test_a_worker_compressed_reset_chat_lists_once(db):
    db.end_session("origin", "session_reset")
    db.create_session("r1", source="cli", parent_session_id="origin", model_config={"_reset_from": "origin"})
    db.append_message("r1", role="user", content="after /new")
    _worker_compress(db, "r1", "r2")
    _worker_compress(db, "r2", "r3")

    rows = db.list_sessions_rich(limit=50, order_by_last_active=True, compact_rows=True, include_pinned=True)
    assert sorted(row["id"] for row in rows) == ["origin", "r3"]
    assert db.session_count(exclude_children=True) == 2
    assert "_reset_from" not in db.get_session("r3")
    assert _config(db, "r1")["_reset_from"] == "origin"
    assert _config(db, "r2") == _config(db, "r3") == {"max_iterations": 90}


def test_a_worker_compressed_delegate_keeps_its_owner(db):
    db.create_session("d1", source="cli", parent_session_id="origin", model_config={"_delegate_from": "origin"})
    db.append_message("d1", role="user", content="delegate goal")
    # The child agent's own model_config carries the marker (tools/delegate_tool.py).
    _worker_compress(db, "d1", "d2", model_config={"_delegate_from": "origin"})
    _worker_compress(db, "d2", "d3")

    assert _config(db, "d2")["_delegate_from"] == _config(db, "d3")["_delegate_from"] == "origin"
    rows = db.list_sessions_rich(limit=50, order_by_last_active=True, compact_rows=True)
    assert [row["id"] for row in rows] == ["origin"]


@pytest.mark.parametrize("marker", ["_branched_from", "_reset_from", "_delegate_from"])
def test_a_worker_cannot_name_another_parent(db, marker):
    db.create_session("c1", source="cli", parent_session_id="origin", model_config={marker: "origin"})
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        _worker_compress(db, "c1", "c2", model_config={marker: "someone-else"})
