"""Tests: the curator run drops usage records whose skill is neither installed nor archived.

Past purges and hub uninstalls left the deleted skill's record in skills/.usage.json. Those records are in users'
files, and a later skill of the same name inherits one (``created_by: agent`` included). The curator's run drops them and names them; a dry run only names them. A record with no lifecycle data
(counters only) is left alone: it can belong to a project or plugin skill the curator cannot list.
"""

import logging
import threading

import pytest

from tools import skill_usage


def _skill(name, *, under="", archived=False):
    root = skill_usage._archive_dir() if archived else skill_usage._skills_dir()
    skill_md = root / under / name / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text(f"---\nname: {name}\ndescription: x\n---\n# body\n", encoding="utf-8")


@pytest.fixture
def curator(monkeypatch):
    from agent import curator
    monkeypatch.setattr(curator, "_load_config", dict)
    monkeypatch.setattr(curator, "_run_llm_review", lambda prompt: "llm-stub")
    yield curator
    for t in threading.enumerate():
        if t.name == "curator-review" and t.is_alive():
            t.join(timeout=10.0)


@pytest.fixture
def records():
    """Records for skills that are gone (purged, uninstalled, deleted) and for skills that are still there."""
    _skill("live")
    _skill("nested-live", under="category")
    _skill("in-archive", archived=True)
    _skill("in-nested-archive", under="old-category", archived=True)
    (skill_usage._skills_dir() / ".hub").mkdir()
    (skill_usage._skills_dir() / ".hub" / "lock.json").write_text('{"installed": {"hub-listed": {}}}')
    usage = {
        "purged-before-77": {"created_by": "agent", "state": skill_usage.STATE_ARCHIVED},
        "uninstalled-before-76": {"created_by": "installed", "state": skill_usage.STATE_ACTIVE},
        "deleted-by-hand": {"created_by": "learn", "state": skill_usage.STATE_ACTIVE},
        "pinned-gone": {"created_by": None, "pinned": True},
        "counters-only": {"created_by": None, "use_count": 3},
        "plugin:tool-skill": {"created_by": "agent"},
        "live": {"created_by": "agent", "state": skill_usage.STATE_ACTIVE},
        "nested-live": {"created_by": "learn"},
        "in-archive": {"created_by": "agent", "state": skill_usage.STATE_ARCHIVED},
        "in-nested-archive": {"created_by": "agent", "state": skill_usage.STATE_ARCHIVED},
        "hub-listed": {"created_by": "installed"},
    }
    assert skill_usage.save_usage(usage)
    return usage


GONE = ["deleted-by-hand", "pinned-gone", "purged-before-77", "uninstalled-before-76"]


def test_the_curator_run_drops_records_of_skills_gone_and_names_them(curator, records, caplog):
    with caplog.at_level(logging.INFO, logger="agent.curator"):
        result = curator.run_curator_review(synchronous=True, consolidate=False)

    assert sorted(skill_usage.load_usage()) == sorted(set(records) - set(GONE))
    assert f"{len(GONE)} stale usage records dropped" in result["summary_so_far"]
    assert any(all(name in r.getMessage() for name in GONE) for r in caplog.records)


def test_a_dry_run_names_them_and_drops_nothing(curator, records, caplog):
    with caplog.at_level(logging.INFO, logger="agent.curator"):
        result = curator.run_curator_review(synchronous=True, consolidate=False, dry_run=True)

    assert skill_usage.load_usage() == records
    assert f"{len(GONE)} stale usage records dropped" in result["summary_so_far"]
    assert any(all(name in r.getMessage() for name in GONE) for r in caplog.records)
