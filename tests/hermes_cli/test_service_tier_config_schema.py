"""The config UI offers every fast-mode tier the runtime reads.

``agent.service_tier`` lacked ``ultrafast``, a tier ``agent.fast_mode`` parses and sends,
so no Settings screen built from the schema could pick it.
"""

from agent.fast_mode import NORMAL_TIER_WORDS, SERVICE_TIER_WORDS, parse_service_tier
from hermes_cli.web_server_config import CONFIG_SCHEMA


def test_service_tier_choices_reach_every_tier():
    field = CONFIG_SCHEMA["agent.service_tier"]
    assert field["type"] == "select"
    reached = {parse_service_tier(option) for option in field["options"]}
    assert sorted(set(SERVICE_TIER_WORDS.values()) - reached) == []


def test_service_tier_choices_are_words_the_runtime_reads():
    field = CONFIG_SCHEMA["agent.service_tier"]
    unknown = [o for o in field["options"] if o not in NORMAL_TIER_WORDS and o not in SERVICE_TIER_WORDS]
    assert unknown == []
