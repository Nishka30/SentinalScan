"""Phase 0: settings come from the environment, and a missing key is not fatal."""

from sentinel.config import Settings


def test_defaults_when_no_env_is_set(monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("SENTINEL_LLM_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.nvidia_api_key is None
    assert settings.llm_enabled is False
    assert settings.llm_model == "meta/llama-3.3-70b-instruct"
    assert settings.llm_base_url == "https://integrate.api.nvidia.com/v1"


def test_env_vars_override_defaults(monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-not-a-real-key")
    monkeypatch.setenv("SENTINEL_LLM_MODEL", "openai/gpt-oss-120b")

    settings = Settings(_env_file=None)

    assert settings.llm_enabled is True
    assert settings.llm_model == "openai/gpt-oss-120b"
