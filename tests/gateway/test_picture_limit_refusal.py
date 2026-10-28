"""A prompt over the picture limit is refused by name, and a client reads the limit first.

Eleven pictures came back as a bare ``invalid_params`` (logged at debug only), the limit was kept
twice (``prompt.submit`` attachments and API image parts), and no answer a client reads carried it,
so a client had to copy the number.
"""
import asyncio
import base64
import json

import pytest

from gateway import session_ingress_media as media
from hermes_state_runtime import RuntimeStoreError

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=')


def _staged(count):
    from gateway.platforms.base import get_image_cache_dir
    staging = get_image_cache_dir()
    paths = [staging / f'picture-{index}.png' for index in range(count)]
    for index, path in enumerate(paths):
        path.write_bytes(PNG + bytes([index]))
    return [{'path': str(path), 'mime': 'image/png'} for path in paths]


def _assert_names_the_limit(refusal, count):
    from gateway.session_controls import _refusal_data
    assert refusal.reason == 'too_many_attachments', refusal
    data = _refusal_data(refusal)
    assert data['field'] == 'attachments' and data['detail'], data
    assert (data['limit'], data['count']) == (media.ATTACHMENT_LIMIT, count), data


def test_prompt_attachments_over_the_limit_are_refused_with_limit_and_count():
    limit = media.ATTACHMENT_LIMIT
    assert media.admit_attachments(_staged(limit))['attachments_v1']['media']  # the limit itself is taken
    with pytest.raises(RuntimeStoreError) as caught:
        media.admit_attachments(_staged(limit + 1))
    _assert_names_the_limit(caught.value, limit + 1)


def test_api_image_parts_share_the_prompt_limit_and_stay_a_400():
    pytest.importorskip('aiohttp')
    from gateway.platforms.api_server import _openai_error
    from gateway.platforms.api_server_authority_runs import store_error_response
    from gateway.session_api_media import commit_api_images
    part = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(PNG).decode()}}
    count = media.ATTACHMENT_LIMIT + 1
    with pytest.raises(RuntimeStoreError) as caught:
        commit_api_images([part] * count)
    _assert_names_the_limit(caught.value, count)
    response = store_error_response(_openai_error, caught.value)
    assert response.status == 400
    assert json.loads(response.body)['error']['code'] == 'too_many_attachments'


def test_the_refusal_is_logged_above_debug(caplog):
    from gateway.session_controls import _log_refusal
    with pytest.raises(RuntimeStoreError) as caught:
        media.check_attachment_count(media.ATTACHMENT_LIMIT + 1)
    with caplog.at_level('DEBUG', logger='gateway.session_controls'):
        _log_refusal('default', 'prompt.submit', caught.value)
    record = caplog.records[-1]
    assert record.levelname == 'WARNING' and 'too_many_attachments' in record.getMessage(), record


@pytest.mark.platforms("linux")
def test_a_client_reads_the_limit_before_sending_and_the_refusal_names_it(tmp_path):
    """Over a real gateway socket: ``session.create``, ``session.resume`` and ``runtime.describe``
    answer the limit, and a prompt with one picture more is refused with the numbers in
    ``error.data``."""
    from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
    from tests.gateway.test_session_busy_controls import owner
    with owner(tmp_path) as (home, _peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                created = await rpc(ws, 'session.create', request_id='picture-limit', source='tui', toolsets=[])
                limit = created['result']['attachment_limit']
                sid = created['result']['session_id']
                assert (await rpc(ws, 'session.resume', session_id=sid))['result']['attachment_limit'] == limit
                assert (await rpc(ws, 'runtime.describe'))['result']['attachment_limit'] == limit
                staging = home / 'cache' / 'images'
                pictures = [{'path': str(staging / f'p{index}.png'), 'mime': 'image/png'}
                            for index in range(limit + 1)]
                refused = await rpc(ws, 'prompt.submit', session_id=sid, submission_id='too-many',
                                    text='look', attachments=pictures)
                error = refused['error']
                assert error['message'] == 'too_many_attachments', refused
                assert error['data']['reason'] == 'too_many_attachments'
                assert (error['data']['limit'], error['data']['count']) == (limit, limit + 1), error
        asyncio.run(probe())
