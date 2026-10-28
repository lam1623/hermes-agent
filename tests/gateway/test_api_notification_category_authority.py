"""A request's notification category reaches its turn under session authority.

``POST /v1/chat/completions`` takes ``hermes_notification_category: "diagnostic"`` from a client that
names its session, and ``run_internal_session_turn`` takes the category of an in-process wake. Without
session authority ``_run_agent`` mutes such a turn when the policy hides diagnostics, and the client gets
no text. Under session authority the admitted turn must carry the same category: the runner mutes it, and
the API door's result says so, so the reply text stays off the wire.
"""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.session_api_turn import admit_api_turn
from gateway.warning_notifications import diagnostic_metadata

KEY = 'api-notification-category-key-long-enough'
DIAGNOSTIC = {'notification_category': 'diagnostic'}
REPLY = 'private diagnostic evidence'


def _turns(owner):
    """What each executed turn's event says: its text, its trust and the category it carries."""
    seen = []

    async def handle(event):
        from gateway.session_results import execution_result
        seen.append({'text': event.text, 'internal': event.internal, 'category': diagnostic_metadata(event)})
        execution_result.get()['result'] = {'final_response': REPLY, 'messages': []}
        return REPLY
    owner.runner._handle_message = handle
    return seen


def _app(api):
    api._api_key = KEY
    app = web.Application()
    app.router.add_post('/v1/chat/completions', api._handle_chat_completions)
    return app


async def _chat(api, **body):
    async with TestClient(TestServer(_app(api))) as client:
        resp = await client.post('/v1/chat/completions', headers={
            'Authorization': 'Bearer ' + KEY, 'X-Hermes-Session-Id': 'client-category'}, json={
            'messages': [{'role': 'user', 'content': 'background diagnostic'}], **body})
        return resp.status, await resp.text()


def _policy(tmp_path, hidden):
    (tmp_path / 'config.yaml').write_text(f'display: {{suppress_warning_notifications: {str(hidden).lower()}}}\n')


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('hidden', [True, False])
async def test_a_clients_diagnostic_category_reaches_its_turn(api, owner, tmp_path, stream, hidden):
    _policy(tmp_path, hidden)
    seen = _turns(owner)
    status, wire = await _chat(api, stream=stream, hermes_notification_category='diagnostic')
    assert status == 200, wire
    # As without session authority: a diagnostic the policy hides gets no text on the wire.
    assert (REPLY in wire) is not hidden
    assert seen == [{'text': 'background diagnostic', 'internal': True, 'category': DIAGNOSTIC}]


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_a_turn_without_the_category_is_never_muted(api, owner, tmp_path, stream):
    _policy(tmp_path, True)
    seen = _turns(owner)
    status, wire = await _chat(api, stream=stream)
    assert status == 200, wire
    assert REPLY in wire
    assert seen == [{'text': 'background diagnostic', 'internal': True, 'category': {}}]


@pytest.mark.asyncio
async def test_an_in_process_diagnostic_wake_keeps_its_category(api, owner):
    seen = _turns(owner)
    await api._run_agent('hello', [], session_id='served-wake')  # the session the wake targets
    await api.run_internal_session_turn(session_id='served-wake', text='wake', profile='work',
                                        notification_category='diagnostic')
    assert [turn['category'] for turn in seen] == [{}, DIAGNOSTIC]


def test_the_admission_records_the_category_for_its_turn(api, owner):
    from gateway.session_api_turn import prepare_api_execution
    _, ref, row = admit_api_turn(api, session_id='client-category', user_message='hello', conversation_history=[],
                                 notification_category='diagnostic')
    assert row['payload']['api_turn_v1']['notification_category'] == 'diagnostic'
    assert prepare_api_execution(owner, ref, row['payload'])['category'] == DIAGNOSTIC
    _, ref, row = admit_api_turn(api, session_id='result-category', user_message='hello', conversation_history=[],
                                 notification_category='result')
    assert 'notification_category' not in row['payload']['api_turn_v1']
    assert prepare_api_execution(owner, ref, row['payload'])['category'] == {}


@pytest.mark.parametrize('category', ['result', 'urgent', None, ['diagnostic']])
def test_an_admission_with_a_malformed_category_is_refused(api, owner, category):
    from gateway.session_api_turn import check_api_turn
    from hermes_state_runtime import RuntimeStoreError
    _, ref, row = admit_api_turn(api, session_id='malformed-category', user_message='hello', conversation_history=[])
    payload = {**row['payload'], 'api_turn_v1': {**row['payload']['api_turn_v1'], 'notification_category': category}}
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        check_api_turn(owner, ref, payload)
