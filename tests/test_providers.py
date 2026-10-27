from __future__ import annotations

import json

import httpx
import pytest

from config import ProviderSettings
from models import ChatMessage
from providers import (
    AnthropicProvider,
    CompatibleProvider,
    GeminiProvider,
    OpenAIProvider,
    ProviderError,
)


def response_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_sends_history_and_returns_actual_response_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["authorization"] = request.headers["Authorization"]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "A real model answer."}}]}
        )

    client = response_client(handler)
    provider = OpenAIProvider(
        ProviderSettings("openai", "secret", "mock-model", "https://api.example/v1"),
        client=client,
    )
    response = provider.send_message(
        "What is the tide?",
        [ChatMessage(role="user", content="Hello"), ChatMessage(role="assistant", content="Hi.")],
    )

    assert response == "A real model answer."
    assert captured["authorization"] == "Bearer secret"
    assert captured["body"]["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": "What is the tide?"},
    ]
    client.close()


def test_gemini_request_uses_provider_role_and_extracts_response():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "Gemini says hello."}]}}]}
        )

    client = response_client(handler)
    provider = GeminiProvider(
        ProviderSettings("gemini", "secret", "gemini-test", "https://api.example/v1beta"),
        client=client,
    )
    assert (
        provider.send_message("Continue", [ChatMessage(role="assistant", content="Prior answer")])
        == "Gemini says hello."
    )
    assert "key=secret" in captured["url"]
    assert captured["body"]["contents"][0]["role"] == "model"
    assert captured["body"]["contents"][1]["parts"][0]["text"] == "Continue"
    client.close()


def test_anthropic_request_sends_required_headers_and_message_history():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": "Claude says hello."}]}
        )

    client = response_client(handler)
    provider = AnthropicProvider(
        ProviderSettings("anthropic", "secret", "claude-test", "https://api.example/v1"),
        client=client,
    )
    assert provider.send_message("Hi", []) == "Claude says hello."
    assert captured["headers"]["x-api-key"] == "secret"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["max_tokens"] == 2048
    client.close()


def test_compatible_endpoint_can_run_without_an_api_key():
    client = response_client(
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": "Compatible response."}}]}
        )
    )
    provider = CompatibleProvider(
        ProviderSettings("compatible", "", "local-model", "https://local.example/v1"),
        client=client,
    )
    assert provider.send_message("Hello", []) == "Compatible response."
    client.close()


def test_compatible_endpoint_supports_azure_style_api_key_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["api_key"] = request.headers.get("api-key")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Azure-compatible response."}}]}
        )

    client = response_client(handler)
    provider = CompatibleProvider(
        ProviderSettings(
            "compatible",
            "azure-secret",
            "deployment-name",
            "https://resource.example/openai/v1",
            "api-key",
            "",
        ),
        client=client,
    )

    assert provider.send_message("Hello", []) == "Azure-compatible response."
    assert captured["api_key"] == "azure-secret"
    client.close()


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "API key"),
        (429, "quota"),
        (503, "temporarily unavailable"),
    ],
)
def test_provider_http_failures_are_explained_without_exposing_body(status_code, message):
    client = response_client(
        lambda request: httpx.Response(status_code, text="private upstream details")
    )
    provider = OpenAIProvider(
        ProviderSettings("openai", "do-not-leak", "test-model", "https://api.example/v1"),
        client=client,
    )

    with pytest.raises(ProviderError, match=message) as error:
        provider.send_message("Hello", [])

    assert "private upstream details" not in str(error.value)
    assert "do-not-leak" not in str(error.value)
    client.close()


def test_provider_rejects_empty_api_response():
    client = response_client(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})
    )
    provider = OpenAIProvider(
        ProviderSettings("openai", "secret", "test-model", "https://api.example/v1"),
        client=client,
    )
    with pytest.raises(ProviderError, match="empty response"):
        provider.send_message("Hello", [])
    client.close()


def test_openai_joins_content_parts_and_falls_back_to_refusal():
    content_client = response_client(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "part one"},
                                {"type": "text", "text": "part two"},
                            ]
                        }
                    }
                ]
            },
        )
    )
    refusal_client = response_client(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": None, "refusal": "I cannot help."}}]},
        )
    )
    settings = ProviderSettings("openai", "secret", "test-model", "https://api.example/v1")
    content_provider = OpenAIProvider(settings, client=content_client)
    refusal_provider = OpenAIProvider(settings, client=refusal_client)

    assert content_provider.send_message("Hello", []) == "part one\npart two"
    assert refusal_provider.send_message("Hello", []) == "I cannot help."
    content_client.close()
    refusal_client.close()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": "not-an-object"}]}),
    ],
)
def test_openai_rejects_unexpected_shapes(response):
    client = response_client(lambda request: response)
    provider = OpenAIProvider(
        ProviderSettings("openai", "secret", "test-model", "https://api.example/v1"),
        client=client,
    )

    with pytest.raises(ProviderError, match="unexpected response format"):
        provider.send_message("Hello", [])
    client.close()


def test_gemini_blocked_requests_are_reported_without_prompt_text():
    client = response_client(
        lambda request: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    )
    provider = GeminiProvider(
        ProviderSettings("gemini", "secret", "gemini-test", "https://api.example/v1beta"),
        client=client,
    )

    with pytest.raises(ProviderError, match=r"blocked the request \(SAFETY\)"):
        provider.send_message("private prompt", [])
    client.close()


def test_gemini_missing_candidates_is_reported():
    client = response_client(lambda request: httpx.Response(200, json={}))
    provider = GeminiProvider(
        ProviderSettings("gemini", "secret", "gemini-test", "https://api.example/v1beta"),
        client=client,
    )

    with pytest.raises(ProviderError, match="unexpected response format"):
        provider.send_message("Hello", [])
    client.close()


def test_anthropic_joins_text_blocks_and_uses_configured_max_tokens():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "tool_use", "name": "ignored"},
                    {"type": "text", "text": "second"},
                ]
            },
        )

    client = response_client(handler)
    provider = AnthropicProvider(
        ProviderSettings(
            "anthropic",
            "secret",
            "claude-test",
            "https://api.example/v1",
            max_tokens=1234,
        ),
        client=client,
    )

    assert provider.send_message("Hello", []) == "first\nsecond"
    assert captured["body"]["max_tokens"] == 1234
    client.close()


def test_compatible_endpoint_omits_authorization_without_a_key():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("Authorization")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Compatible response."}}]}
        )

    client = response_client(handler)
    provider = CompatibleProvider(
        ProviderSettings("compatible", "", "local-model", "https://local.example/v1"),
        client=client,
    )

    assert provider.send_message("Hello", []) == "Compatible response."
    assert captured["authorization"] is None
    client.close()


def test_compatible_endpoint_rejects_a_blank_header_name_when_a_key_is_set():
    with pytest.raises(ValueError, match="API_KEY_HEADER"):
        CompatibleProvider(
            ProviderSettings(
                "compatible",
                "secret",
                "local-model",
                "https://local.example/v1",
                api_key_header=" ",
            )
        )


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (httpx.ReadTimeout, "timed out"),
        (httpx.ConnectError, "Could not connect"),
    ],
)
def test_provider_network_failures_are_explained(exception, message):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception("network failure", request=request)

    client = response_client(handler)
    provider = OpenAIProvider(
        ProviderSettings("openai", "secret", "test-model", "https://api.example/v1"),
        client=client,
    )

    with pytest.raises(ProviderError, match=message):
        provider.send_message("Hello", [])
    client.close()


def test_provider_rejects_invalid_json_and_non_object_json():
    invalid_json_client = response_client(lambda request: httpx.Response(200, text="not json"))
    list_json_client = response_client(lambda request: httpx.Response(200, json=["nope"]))
    settings = ProviderSettings("openai", "secret", "test-model", "https://api.example/v1")

    with pytest.raises(ProviderError, match="invalid JSON"):
        OpenAIProvider(settings, client=invalid_json_client).send_message("Hello", [])
    with pytest.raises(ProviderError, match="unexpected response format"):
        OpenAIProvider(settings, client=list_json_client).send_message("Hello", [])
    invalid_json_client.close()
    list_json_client.close()


def test_client_errors_are_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="private upstream details")

    client = response_client(handler)
    provider = OpenAIProvider(
        ProviderSettings("openai", "secret", "test-model", "https://api.example/v1"),
        client=client,
        retries=3,
        retry_delay=0,
    )

    with pytest.raises(ProviderError, match="API key"):
        provider.send_message("Hello", [])
    assert calls == 1
    client.close()


def test_transient_server_errors_are_retried_with_backoff():
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Recovered."}}]})

    client = response_client(handler)
    provider = OpenAIProvider(
        ProviderSettings("openai", "secret", "test-model", "https://api.example/v1"),
        client=client,
        retries=2,
        retry_delay=0.25,
        sleep=sleeps.append,
    )

    assert provider.send_message("Hello", []) == "Recovered."
    assert calls == 2
    assert sleeps == [0.25]
    client.close()


def test_rate_limit_retry_after_is_respected():
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Recovered."}}]})

    client = response_client(handler)
    provider = OpenAIProvider(
        ProviderSettings("openai", "secret", "test-model", "https://api.example/v1"),
        client=client,
        retries=1,
        retry_delay=0,
        sleep=sleeps.append,
    )

    assert provider.send_message("Hello", []) == "Recovered."
    assert sleeps == [2.0]
    client.close()


def test_retries_stop_after_the_configured_attempts():
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    client = response_client(handler)
    provider = OpenAIProvider(
        ProviderSettings("openai", "secret", "test-model", "https://api.example/v1"),
        client=client,
        retries=2,
        retry_delay=0.25,
        sleep=sleeps.append,
    )

    with pytest.raises(ProviderError, match="temporarily unavailable"):
        provider.send_message("Hello", [])
    assert calls == 3
    assert sleeps == [0.25, 0.5]
    client.close()


def test_timeouts_are_retried_before_succeeding():
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("slow response", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Recovered."}}]})

    client = response_client(handler)
    provider = OpenAIProvider(
        ProviderSettings("openai", "secret", "test-model", "https://api.example/v1"),
        client=client,
        retries=1,
        retry_delay=0,
        sleep=sleeps.append,
    )

    assert provider.send_message("Hello", []) == "Recovered."
    assert calls == 2
    assert sleeps == [0.0]
    client.close()


def test_negative_retry_configuration_is_rejected():
    settings = ProviderSettings("openai", "secret", "test-model", "https://api.example/v1")

    with pytest.raises(ValueError, match="retries cannot be negative"):
        OpenAIProvider(settings, retries=-1)
    with pytest.raises(ValueError, match="retry delay cannot be negative"):
        OpenAIProvider(settings, retry_delay=-1)


def test_owned_clients_are_closed_but_injected_clients_are_not():
    settings = ProviderSettings("openai", "secret", "test-model", "https://api.example/v1")
    owned = OpenAIProvider(settings)
    owned_client = owned._client
    owned.close()
    assert owned_client.is_closed

    injected_client = response_client(lambda request: httpx.Response(200, json={"choices": []}))
    injected = OpenAIProvider(settings, client=injected_client)
    injected.close()
    assert not injected_client.is_closed
    injected_client.close()
