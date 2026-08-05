"""Provider selection: which vendor, which model string, whose key.

Nothing here talks to a model. The point of llm.py is that the vendor question
is decided in plain Python before a single token is spent, so it can be tested
the same way.
"""

import pytest

from stocks_crew import llm


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from "nothing configured"."""
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "OPENAI_MODEL", "GEMINI_MODEL", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Model strings — CrewAI routes on the "<provider>/" prefix, so it is not
# cosmetic; it is the whole mechanism by which a run lands on one vendor.
# ---------------------------------------------------------------------------


def test_default_models_per_provider():
    assert llm.model_for("openai") == "openai/gpt-4o-mini"
    assert llm.model_for("gemini") == "gemini/gemini-2.5-flash"


def test_model_env_overrides_the_default(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    assert llm.model_for("openai") == "openai/gpt-4.1-mini"


def test_a_bare_gemini_model_gets_the_routing_prefix(monkeypatch):
    """Unprefixed, gemini-2.5-pro is taken for an OpenAI model name."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    assert llm.model_for("gemini") == "gemini/gemini-2.5-pro"


def test_a_fully_qualified_gemini_model_is_left_alone(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini/gemini-2.5-pro")
    assert llm.model_for("gemini") == "gemini/gemini-2.5-pro"


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_every_default_model_is_one_crewai_can_actually_route(provider):
    """The prefixed form is only useful if CrewAI knows the model behind it.

    An unknown model part makes CrewAI fall through to LiteLLM, which is not
    installed — so this is the test that catches a stale default before a
    student's first run does.
    """
    from crewai.llms.constants import GEMINI_MODELS, OPENAI_MODELS

    known = {"openai": OPENAI_MODELS, "gemini": GEMINI_MODELS}[provider]
    assert llm.model_for(provider).split("/", 1)[1] in known


def test_an_unknown_provider_is_refused():
    with pytest.raises(llm.UnknownProvider):
        llm.get_provider("llama-at-home")


# ---------------------------------------------------------------------------
# Which provider a run lands on
# ---------------------------------------------------------------------------


def test_with_nothing_set_the_default_is_openai():
    assert llm.default_provider() == "openai"


def test_the_provider_with_a_key_wins(monkeypatch):
    """Dropping GEMINI_API_KEY alone into .env is enough to run on Gemini."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    assert llm.default_provider() == "gemini"


def test_llm_provider_beats_the_keys_that_happen_to_be_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert llm.default_provider() == "gemini"


def test_configured_providers_reports_only_the_ones_with_keys(monkeypatch):
    assert llm.configured_providers() == []
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    assert llm.configured_providers() == ["gemini"]


# ---------------------------------------------------------------------------
# resolve() — provider + model + key, or a typed failure
# ---------------------------------------------------------------------------


def test_resolve_takes_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-env")
    config = llm.resolve("gemini")
    assert (config.name, config.model, config.api_key) == (
        "gemini",
        "gemini/gemini-2.5-flash",
        "AIza-env",
    )


@pytest.mark.parametrize(
    "provider,client",
    [("openai", "OpenAICompletion"), ("gemini", "GeminiCompletion")],
)
def test_the_resolved_model_builds_that_vendors_client(provider, client):
    """End of the chain: provider name in, vendor SDK client out. No call made.

    Built inside provider_environment() because these clients read the key at
    construction time — which is the whole reason that context manager exists.
    """
    from crewai import LLM

    config = llm.resolve(provider, "test-key")
    with llm.provider_environment(config):
        assert type(LLM(model=config.model)).__name__ == client


def test_a_per_request_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert llm.resolve("openai", "sk-sidebar").api_key == "sk-sidebar"


def test_a_key_for_one_provider_does_not_unlock_the_other(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    with pytest.raises(llm.MissingAPIKey) as excinfo:
        llm.resolve("gemini")
    assert excinfo.value.error_code == "gemini_api_key_required"


def test_the_missing_key_error_names_the_variable_to_set():
    with pytest.raises(llm.MissingAPIKey) as excinfo:
        llm.resolve("openai")
    assert "OPENAI_API_KEY" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The environment handoff — the only way the vendor SDK ever sees the key
# ---------------------------------------------------------------------------


def test_the_key_is_exported_for_the_run_and_restored_after(monkeypatch):
    import os

    monkeypatch.setenv("GEMINI_API_KEY", "AIza-env")
    config = llm.resolve("gemini", "AIza-sidebar")
    with llm.provider_environment(config):
        assert os.environ["GEMINI_API_KEY"] == "AIza-sidebar"
    assert os.environ["GEMINI_API_KEY"] == "AIza-env"


def test_a_sidebar_key_does_not_linger_when_the_env_had_none():
    import os

    config = llm.resolve("gemini", "AIza-sidebar")
    with llm.provider_environment(config):
        assert os.environ["GEMINI_API_KEY"] == "AIza-sidebar"
    assert "GEMINI_API_KEY" not in os.environ
