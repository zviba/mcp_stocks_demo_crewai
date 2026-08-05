# llm.py — which model the crew talks to, and with whose key.
#
# CrewAI picks a vendor SDK from the *model string* you hand `LLM(...)`, written
# in LiteLLM's "<provider>/<model>" syntax: "openai/gpt-4o-mini" builds an
# OpenAICompletion that reads OPENAI_API_KEY, "gemini/gemini-2.5-flash" builds a
# GeminiCompletion that reads GEMINI_API_KEY. The key is not something we can
# pass down to the agents — it is read from the environment at construction
# time. So the whole provider question reduces to two decisions, and this module
# is where both are made:
#
#   1. which model string to build the LLM with;
#   2. which environment variable to put the key in while the crew runs.
#
# Keeping that here means crew.py never mentions a vendor, and adding a third
# provider is one entry in PROVIDERS rather than a hunt through the codebase.
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional


@dataclass(frozen=True)
class Provider:
    """Everything that differs between one model vendor and the next."""

    name: str  # what the API/UI pass around
    label: str  # what a human sees
    api_key_env: str  # where the vendor SDK looks for the key
    model_env: str  # optional override of the model
    default_model: str  # "<provider>/<model>", LiteLLM syntax
    key_prefix: str  # the "<provider>/" CrewAI routes on
    error_code: str  # in-band error when no key is available


PROVIDERS: Dict[str, Provider] = {
    "openai": Provider(
        name="openai",
        label="OpenAI",
        api_key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        default_model="openai/gpt-4o-mini",
        key_prefix="openai/",
        error_code="openai_api_key_required",
    ),
    "gemini": Provider(
        name="gemini",
        label="Google Gemini",
        api_key_env="GEMINI_API_KEY",
        model_env="GEMINI_MODEL",
        # "gemini/" routes to CrewAI's native Gemini client (google-genai,
        # pulled in by the crewai[google-genai] extra), which uses an API key
        # when it has one and only falls back to Vertex AI / GCP credentials
        # when GOOGLE_CLOUD_PROJECT is set instead.
        default_model="gemini/gemini-2.5-flash",
        key_prefix="gemini/",
        error_code="gemini_api_key_required",
    ),
}

# What you get when nobody says otherwise and no key is set anywhere — which
# also decides *which* "key required" error you get told about.
DEFAULT_PROVIDER = "openai"


class MissingAPIKey(Exception):
    """No usable key for the chosen provider. Carries the in-band error code."""

    def __init__(self, provider: Provider):
        self.provider = provider
        self.error_code = provider.error_code
        super().__init__(
            f"{provider.label} API key is required to run the crew. Set "
            f"{provider.api_key_env} in .env or enter one in the sidebar."
        )


class UnknownProvider(Exception):
    """A provider name that is not in PROVIDERS."""

    error_code = "unknown_provider"


@dataclass(frozen=True)
class LLMConfig:
    """The resolved answer: this provider, this model, this key."""

    provider: Provider
    model: str
    api_key: str

    @property
    def name(self) -> str:
        return self.provider.name


def provider_names() -> List[str]:
    """Every provider this demo knows how to talk to."""
    return list(PROVIDERS)


def get_provider(name: Optional[str]) -> Provider:
    """Look up a provider by name, case- and whitespace-insensitively."""
    key = (name or "").strip().lower()
    if not key:
        return PROVIDERS[DEFAULT_PROVIDER]
    try:
        return PROVIDERS[key]
    except KeyError:
        raise UnknownProvider(
            f"Unknown provider {name!r}. Known providers: {sorted(PROVIDERS)}"
        ) from None


def configured_providers() -> List[str]:
    """Providers that have a key in the environment right now.

    The UI uses this to say "Gemini is ready, OpenAI is not" without asking the
    user to test-drive a run first.
    """
    return [p.name for p in PROVIDERS.values() if (os.getenv(p.api_key_env) or "").strip()]


def model_for(provider: Optional[str] = None) -> str:
    """The "<provider>/<model>" string for a provider, honouring its *_MODEL override.

    A bare override is prefixed for you, so GEMINI_MODEL=gemini-2.5-pro works as
    well as the fully-qualified GEMINI_MODEL=gemini/gemini-2.5-pro. Unprefixed,
    an unrecognised name is assumed to be OpenAI's, which fails in a confusing
    place — an auth error against the wrong vendor.

    One caveat that comes with the prefixed form: CrewAI checks the model part
    against its own list of known models per vendor, and a name it does not know
    falls through to LiteLLM, which this project does not install. If you set a
    model newer than the installed crewai, upgrade crewai rather than guessing.
    """
    p = get_provider(provider)
    model = (os.getenv(p.model_env) or "").strip() or p.default_model
    if p.key_prefix and "/" not in model:
        model = f"{p.key_prefix}{model}"
    return model


def default_provider() -> str:
    """Which provider a request gets when it does not name one.

    LLM_PROVIDER wins if set; otherwise the first provider that actually has a
    key, so dropping GEMINI_API_KEY alone into .env is enough to run on Gemini.
    """
    declared = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if declared:
        return get_provider(declared).name
    for name in PROVIDERS:
        if (os.getenv(PROVIDERS[name].api_key_env) or "").strip():
            return name
    return DEFAULT_PROVIDER


def resolve(provider: Optional[str] = None, api_key: str = "") -> LLMConfig:
    """Pick a provider and a key, or raise.

    A per-request key beats the environment, which is what lets the Streamlit
    sidebar and the /analyze body override .env without touching it.
    """
    p = get_provider(provider or default_provider())
    key = (api_key or "").strip() or (os.getenv(p.api_key_env) or "").strip()
    if not key:
        raise MissingAPIKey(p)
    return LLMConfig(provider=p, model=model_for(p.name), api_key=key)


@contextmanager
def provider_environment(config: LLMConfig) -> Iterator[None]:
    """Put the key where the vendor SDK will find it, for the duration of the run.

    Restores the previous value on the way out, so a key typed into the sidebar
    for one run does not silently become the key for the next one.
    """
    var = config.provider.api_key_env
    previous = os.environ.get(var)
    os.environ[var] = config.api_key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = previous
