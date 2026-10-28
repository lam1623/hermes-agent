"""A managed Nous tool row the signed-in plan does not include is refused before anything is written.

``PUT /api/tools/toolsets/{name}/provider`` (the Desktop's provider picker) wrote ``tts.provider: nous``
and only then asked the Portal, so a plan without TTS replaced a working Edge TTS with a provider that
never activates. Signing in cannot fix a plan, so that pick is refused and the previous provider stays.
A signed-out pick still writes and reports ``needs_nous_auth`` (the Desktop finishes it with a
sign-in), and the CLI already gated first.
"""

import pytest

from hermes_cli.nous_account import NousPortalAccountInfo, NousToolAccessInfo


def _account(**kwargs):
    base = dict(logged_in=True, source="account_api", fresh=True, portal_base_url="https://portal.example")
    return NousPortalAccountInfo(**{**base, **kwargs})


# A pool account whose live pool funds image generation but not TTS, and an account without paid access.
POOL_WITHOUT_TTS = _account(paid_service_access=False,
                            tool_access=NousToolAccessInfo(enabled=True, coverage={"fal-image": True, "tts": False}))
NO_PAID_ACCESS = _account(paid_service_access=False)
PAID = _account(paid_service_access=True)
SIGNED_OUT = NousPortalAccountInfo(logged_in=False, source="none", fresh=False, paid_service_access=None)


@pytest.fixture()
def client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    import hermes_state
    from hermes_cli.config import load_config, save_config
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    config = load_config()
    config.setdefault("tts", {})["provider"] = "edge"  # the working choice a refused pick must keep
    save_config(config)
    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def _portal_answers(monkeypatch, account):
    """Stand-in Portal answer at the code's own seam (no real Nous login, no network)."""
    monkeypatch.setattr("hermes_cli.nous_subscription.get_nous_portal_account_info", lambda *a, **k: account)


def _tts_provider():
    from hermes_cli.config import load_config
    return load_config()["tts"]["provider"]


@pytest.mark.parametrize("account", [POOL_WITHOUT_TTS, NO_PAID_ACCESS], ids=["pool-without-tts", "no-paid-access"])
def test_a_nous_row_outside_the_plan_keeps_the_working_provider(client, monkeypatch, account):
    _portal_answers(monkeypatch, account)
    resp = client.put("/api/tools/toolsets/tts/provider", json={"provider": "Nous Subscription"})
    assert _tts_provider() == "edge"
    assert resp.status_code == 403, resp.text
    assert "Nous Subscription" in resp.json()["detail"]


def test_a_nous_row_inside_the_plan_is_written(client, monkeypatch):
    _portal_answers(monkeypatch, PAID)
    resp = client.put("/api/tools/toolsets/tts/provider", json={"provider": "Nous Subscription"})
    assert resp.status_code == 200, resp.text
    assert "needs_nous_auth" not in resp.json()
    assert _tts_provider() == "nous"


def test_a_signed_out_pick_is_written_and_asks_for_sign_in(client, monkeypatch):
    _portal_answers(monkeypatch, SIGNED_OUT)
    resp = client.put("/api/tools/toolsets/tts/provider", json={"provider": "Nous Subscription"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["needs_nous_auth"] is True and resp.json()["feature"] == "tts"
    assert _tts_provider() == "nous"


def test_a_web_capability_outside_the_plan_keeps_its_backend(client, monkeypatch):
    from hermes_cli.config import load_config, save_config
    config = load_config()
    config.setdefault("web", {})["search_backend"] = "searxng"
    save_config(config)
    _portal_answers(monkeypatch, NO_PAID_ACCESS)
    rows = client.get("/api/tools/toolsets/web/config").json()["providers"]
    managed = next(row["name"] for row in rows if row["name"].startswith("Nous Subscription"))
    resp = client.put("/api/tools/toolsets/web/provider", json={"provider": managed, "capability": "search"})
    assert resp.status_code == 403, resp.text
    assert load_config()["web"]["search_backend"] == "searxng"


def test_the_cli_pick_checks_the_plan_before_writing(monkeypatch, capsys):
    """``hermes tools`` runs the Portal gate before its write (``_configure_provider``); same account."""
    import hermes_cli.tools_config_providers as providers
    from hermes_cli.tools_config import TOOL_CATEGORIES

    _portal_answers(monkeypatch, POOL_WITHOUT_TTS)
    row = next(p for p in TOOL_CATEGORIES["tts"]["providers"] if p["name"] == "Nous Subscription")
    config = {"tts": {"provider": "edge"}}
    providers._configure_provider(row, config)
    assert config == {"tts": {"provider": "edge"}}
    assert "isn't included with your current Nous Portal access" in capsys.readouterr().out
