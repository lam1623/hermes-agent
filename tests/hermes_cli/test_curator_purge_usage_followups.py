"""Tests: `hermes curator purge` keeps skills/.usage.json in step with rollback, nested archives and re-creates.

- Rolling a purge back (``hermes curator rollback <entry>``) restores the records the purge dropped with the
  files, so the restored skill is managed again without ``hermes curator adopt``.
- An old nested archive (``.archive/<category>/<skill>/``, from before archives were flattened) holds several
  skills; the purge forgets the record of each skill inside it.
- The check that a record still reads ``archived`` and the forget are one update under the usage file's lock, so a
  skill re-created under that name in between keeps its new record.
"""

import os
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import hermes_cli.curator as curator_cli
from tools import skill_ledger, skill_usage


def _skill(name, folder=None):
    skill_md = skill_usage._skills_dir() / (folder or name) / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text(f"---\nname: {name}\ndescription: x\n---\n# body\n", encoding="utf-8")
    return skill_md.parent


def _age(path, days_ago, *names):
    """``path`` (an archive dir) and the records of ``names`` archived ``days_ago`` days back."""
    when = time.time() - days_ago * 86400
    usage = skill_usage.load_usage()
    for name in names:
        usage[name]["archived_at"] = datetime.fromtimestamp(when, UTC).isoformat()
    assert skill_usage.save_usage(usage)
    os.utime(path, (when, when))


def _archived(name, days_ago=100):
    _skill(name)
    skill_usage.record_created(name, agent_created=True)
    ok, msg = skill_usage.archive_skill(name)
    assert ok, msg
    archive = skill_usage._archive_dir() / name
    _age(archive, days_ago, name)
    return archive


def _purge():
    return curator_cli._cmd_purge(SimpleNamespace(days=30, dry_run=False, yes=True))


def test_rolling_back_a_purge_restores_the_record_with_the_files():
    archive = _archived("old-helper")
    before = skill_usage.load_usage()["old-helper"]
    assert _purge() == 0
    assert "old-helper" not in skill_usage.load_usage()
    (entry,) = [e for e in skill_ledger.list_entries(skill="old-helper") if e.get("action") == "purge"]

    ok, msg = skill_ledger.rollback_entry(entry["id"])

    assert ok, msg
    assert (archive / "SKILL.md").is_file()
    assert skill_usage.load_usage()["old-helper"] == before
    # Restored from the archive, it is the curator's again without `hermes curator adopt`.
    ok, msg = skill_usage.restore_skill("old-helper")
    assert ok, msg
    assert skill_usage.is_curator_managed("old-helper")


def test_rolling_back_a_purge_leaves_a_record_written_since():
    """A skill created under that name after the purge owns the name's record: the rollback restores the files and
    keeps that record."""
    _archived("old-helper")
    assert _purge() == 0
    (entry,) = [e for e in skill_ledger.list_entries(skill="old-helper") if e.get("action") == "purge"]
    _skill("old-helper")
    skill_usage.record_created("old-helper", agent_created=False)

    ok, msg = skill_ledger.rollback_entry(entry["id"])

    assert ok, msg
    assert skill_usage.load_usage()["old-helper"]["created_by"] == "learn"


def test_purging_a_nested_archive_forgets_every_skill_inside_it():
    root = skill_usage._archive_dir() / "mlops"
    for folder, name in (("training/accelerate", "huggingface-accelerate"), ("serving/vllm", "vllm")):
        skill_md = root / folder / "SKILL.md"
        skill_md.parent.mkdir(parents=True)
        skill_md.write_text(f"---\nname: {name}\ndescription: x\n---\n# body\n", encoding="utf-8")
    usage = skill_usage.load_usage()
    for name in ("huggingface-accelerate", "vllm"):
        usage[name] = {"created_by": "agent", "state": skill_usage.STATE_ARCHIVED}
    assert skill_usage.save_usage(usage)
    _age(root, 100, "huggingface-accelerate", "vllm")
    _skill("keeper")
    skill_usage.record_created("keeper", agent_created=True)

    assert _purge() == 0

    assert not root.exists()
    usage = skill_usage.load_usage()
    assert "huggingface-accelerate" not in usage and "vllm" not in usage
    assert usage["keeper"]["created_by"] == "agent"


def test_a_nested_skill_another_archive_still_holds_keeps_its_record():
    root = skill_usage._archive_dir() / "mlops"
    skill_md = root / "serving" / "vllm" / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text("---\nname: vllm\ndescription: x\n---\n# body\n", encoding="utf-8")
    usage = skill_usage.load_usage()
    usage["vllm"] = {"created_by": "agent", "state": skill_usage.STATE_ARCHIVED}
    assert skill_usage.save_usage(usage)
    _age(root, 100, "vllm")
    newer = skill_usage._archive_dir() / "vllm"
    newer.mkdir()
    (newer / "SKILL.md").write_text("---\nname: vllm\ndescription: x\n---\n# body\n", encoding="utf-8")

    assert _purge() == 0

    assert not root.exists() and newer.exists()
    assert skill_usage.load_usage()["vllm"]["state"] == skill_usage.STATE_ARCHIVED


def test_a_skill_re_created_while_the_purge_runs_keeps_its_new_record(monkeypatch):
    """The skill is re-created under the purged name just before the purge takes the usage file's lock to forget.
    The check that the record still reads ``archived`` must be made under that lock, in the same update."""
    _archived("old-helper")
    real_lock = skill_usage._usage_file_lock
    recreated = []

    @contextmanager
    def lock_taken_after_a_re_create():
        if not recreated:
            recreated.append(True)
            usage = skill_usage.load_usage()
            usage["old-helper"] = {**skill_usage._empty_record(), "created_by": "learn"}
            assert skill_usage.save_usage(usage)
        with real_lock():
            yield

    monkeypatch.setattr(skill_usage, "_usage_file_lock", lock_taken_after_a_re_create)

    assert _purge() == 0

    assert recreated
    assert skill_usage.load_usage()["old-helper"]["created_by"] == "learn"
