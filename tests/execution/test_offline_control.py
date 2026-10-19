"""Public offline controls execute participant tools without a provider SDK."""

import json

import pytest

from trialagentbench.adapters.llm_providers import ProviderRouting, get_provider
from trialagentbench.ports.llm_provider import JsonObject


@pytest.mark.parametrize("language,filename", [("python", "analysis.py"), ("r", "analysis.R")])
def test_offline_control_saves_reproducible_script_without_credentials(monkeypatch, language, filename):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = get_provider(
        f"smoke-zero-{language}",
        routing=ProviderRouting("offline_control"),
        send_temperature=False,
        timeout_s=1,
    )
    tools: list[JsonObject] = [{"function": {"name": "execute_code"}}, {"function": {"name": "submit_response"}}]
    response = provider.generate_turn([], tools, temperature=0, max_tokens=100)
    call = response.tool_calls[0]
    assert call.name == "execute_code"
    assert filename in json.loads(call.arguments)["code"]
    assert response.metadata.reported_cost_usd == 0
    assert response.metadata.request_attempts == 1
    messages: list[JsonObject] = [
        {"role": "assistant", "tool_calls": [{"function": {"name": call.name}}]},
        {"role": "tool", "content": "Saved analysis"},
    ]
    response = provider.generate_turn(messages, tools, temperature=0, max_tokens=100)
    assert response.tool_calls[0].name == "submit_response"
    assert "not clinical or model-performance evidence" in response.tool_calls[0].arguments


def test_offline_control_rejects_unknown_models_and_missing_tools():
    with pytest.raises(ValueError, match="Unknown offline"):
        get_provider("real-model", routing=ProviderRouting("offline_control"), send_temperature=False, timeout_s=1)
    provider = get_provider(
        "smoke-zero-python",
        routing=ProviderRouting("offline_control"),
        send_temperature=False,
        timeout_s=1,
    )
    with pytest.raises(ValueError, match="unavailable"):
        provider.generate_turn([], [], temperature=0, max_tokens=100)


def test_offline_study_control_activates_enough_sites_in_a_slow_accrual_world(tmp_path):
    """Late-opening, registry-rate sites cannot fill the study before its look; the control adds sites."""
    import subprocess
    import sys

    from trialagentbench.adapters.offline_control import (
        STUDY_FORECAST_SCREENED,
        STUDY_LAST_ENROLMENT_DAY,
        STUDY_SITE_CANDIDATES,
    )

    provider = get_provider(
        "smoke-zero-python", routing=ProviderRouting("offline_control"), send_temperature=False, timeout_s=1
    )
    tools: list[JsonObject] = [{"function": {"name": name}} for name in ("execute_code", "start_study")]
    code = json.loads(provider.generate_turn([], tools, temperature=0, max_tokens=100).tool_calls[0].arguments)["code"]
    for folder in ("interface", "documents", "scratch"):
        (tmp_path / folder).mkdir()
    (tmp_path / "interface" / "targets.csv").write_text("target_id\naw-1:effect\n")
    # The first two sites list 487 candidates but open on days 113-116 and screen 0.37 a month: about
    # three candidates each before the last useful enrolment day (a release census world).
    sites = [(113, 0.3735, 278), (116, 0.3735, 209), (88, 0.3735, 254), (80, 1.1205, 767), (26, 1.1205, 722)]
    sites += [(48, 0.747, 469)] * 20
    (tmp_path / "documents" / "site_network.csv").write_text(
        "activation_order,site_id,opening_day,screening_candidates_per_month,candidates\n"
        + "".join(
            f"{order},site_{order},{day},{rate},{listed}\n" for order, (day, rate, listed) in enumerate(sites, 1)
        )
    )
    printed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout
    design = json.loads(printed.splitlines()[0])

    def forecast(chosen):
        return sum(rate * max(0.0, STUDY_LAST_ENROLMENT_DAY - day) / 30.4375 for day, rate, _ in chosen)

    chosen = sites[: design["sites"]]
    assert design["sites"] > 2
    assert sum(listed for *_, listed in chosen) >= STUDY_SITE_CANDIDATES
    assert forecast(chosen) >= STUDY_FORECAST_SCREENED > forecast(chosen[:-1])
    assert design["looks"] == [{"deadline_day": 365, "completed_participants": 60}]
