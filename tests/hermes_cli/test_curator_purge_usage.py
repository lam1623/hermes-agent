"""Tests: `hermes curator purge` forgets the skills/.usage.json record of each archived skill it deletes
(hermes_cli/curator.py::_cmd_purge).

The purge deleted archived skills for good and left their records. A later skill of the same name, one the user
writes by hand included, inherited the record, ``created_by: agent`` and all, so the curator took it for its own;
the agent's own delete already forgets. The record goes by the SKILL.md frontmatter
name the purge reads, once the folder is gone; it stays when the removal fails, when another archive of that name
is left, and when it no longer reads ``archived`` (a skill re-created or restored under that name since owns it).
"""

import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import hermes_cli.curator as curator_cli
from tools import skill_usage


def _skill(name, folder=None):
    """A local skill ``name`` in ``skills/<folder or name>/``."""
    skill_md = skill_usage._skills_dir() / (folder or name) / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text(f"---\nname: {name}\ndescription: x\n---\n# body\n", encoding="utf-8")
    return skill_md.parent


def _age(archive, name, days_ago):
    """Archived ``days_ago`` days back: the archive dir's mtime and the record's ``archived_at``."""
    when = time.time() - days_ago * 86400
    os.utime(archive, (when, when))
    usage = skill_usage.load_usage()
    usage[name]["archived_at"] = datetime.fromtimestamp(when, UTC).isoformat()
    assert skill_usage.save_usage(usage)


def _archived(name, days_ago=100):
    """An agent-created skill ``name``, archived ``days_ago`` days back; its archive dir."""
    _skill(name)
    skill_usage.record_created(name, agent_created=True)
    ok, msg = skill_usage.archive_skill(name)
    assert ok, msg
    archive = skill_usage._archive_dir() / name
    _age(archive, name, days_ago)
    return archive


def _purge():
    return curator_cli._cmd_purge(SimpleNamespace(days=30, dry_run=False, yes=True))


def _refuse(path, *args, **kwargs):
    raise PermissionError(13, "Permission denied", str(path))


def test_purge_forgets_the_usage_record():
    archive = _archived("old-helper")
    _skill("keeper")
    skill_usage.record_created("keeper", agent_created=True)

    assert _purge() == 0

    assert not archive.exists()
    usage = skill_usage.load_usage()
    assert "old-helper" not in usage
    assert usage["keeper"]["created_by"] == "agent"
    # A skill the user then writes by hand under that name is not the curator's.
    _skill("old-helper")
    assert not skill_usage.is_curator_managed("old-helper")


def test_the_record_goes_by_the_frontmatter_name():
    """A legacy archive flattened under its folder name (`accelerate` for `huggingface-accelerate`): the record keyed
    by the frontmatter name goes; the record of a live skill named like the folder stays."""
    archive = skill_usage._archive_dir() / "accelerate"
    archive.mkdir(parents=True)
    (archive / "SKILL.md").write_text(
        "---\nname: huggingface-accelerate\ndescription: x\n---\n# body\n", encoding="utf-8")
    _skill("accelerate")
    skill_usage.record_created("accelerate", agent_created=True)
    usage = skill_usage.load_usage()
    usage["huggingface-accelerate"] = {"created_by": "agent", "state": skill_usage.STATE_ARCHIVED}
    assert skill_usage.save_usage(usage)
    _age(archive, "huggingface-accelerate", days_ago=100)

    assert _purge() == 0

    assert not archive.exists()
    usage = skill_usage.load_usage()
    assert "huggingface-accelerate" not in usage
    assert usage["accelerate"]["created_by"] == "agent"


def test_a_failed_removal_keeps_the_record(monkeypatch):
    archive = _archived("old-helper")

    monkeypatch.setattr(shutil, "rmtree", _refuse)
    assert _purge() == 0

    assert (archive / "SKILL.md").is_file()
    assert skill_usage.load_usage()["old-helper"]["state"] == skill_usage.STATE_ARCHIVED


def test_another_archive_of_that_name_keeps_the_record(monkeypatch):
    """Archived twice (`old-helper` and a timestamped `old-helper-<ts>`, one record), both past the TTL: the second
    removal fails, so that archive keeps the record."""
    first = _archived("old-helper")
    _skill("old-helper")
    skill_usage.record_created("old-helper", agent_created=True)
    ok, msg = skill_usage.archive_skill("old-helper")
    assert ok, msg
    (second,) = (p for p in skill_usage._archive_dir().iterdir() if p != first)
    _age(second, "old-helper", days_ago=100)
    rmtree = shutil.rmtree
    monkeypatch.setattr(
        shutil, "rmtree", lambda path, *a, **kw: _refuse(path) if Path(path) == second else rmtree(path, *a, **kw))

    assert _purge() == 0

    assert not first.exists()
    assert (second / "SKILL.md").is_file()
    assert skill_usage.load_usage()["old-helper"]["state"] == skill_usage.STATE_ARCHIVED


def test_the_record_of_a_live_skill_of_that_name_stays():
    """The review fork made the skill again after the archive (a fresh, active record) and the user pinned it:
    purging the old archive leaves that record alone."""
    archive = _archived("old-helper")
    _skill("old-helper")
    skill_usage.record_created("old-helper", agent_created=True)
    assert skill_usage.set_pinned("old-helper", True)

    assert _purge() == 0

    assert not archive.exists()
    record = skill_usage.load_usage()["old-helper"]
    assert (record["created_by"], record["state"], record["pinned"]) == ("agent", skill_usage.STATE_ACTIVE, True)
