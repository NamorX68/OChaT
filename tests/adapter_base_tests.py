"""Tests for `LLMAdapter.get_client()`'s default implementation and `AdapterClientUnavailableError`.

`LLMAdapter` is abstract, so these tests exercise `get_client()` through a minimal concrete
subclass rather than instantiating `LLMAdapter` itself - `send_prompt_async`/`send_prompt_stream`
are irrelevant here and left unimplemented beyond the bare minimum needed to satisfy `ABC`.
"""
from collections.abc import AsyncIterator

import pytest

from ocht.adapters.base import AdapterClientUnavailableError, HealthProbeClient, LLMAdapter


class _MinimalAdapter(LLMAdapter):
    """Concrete `LLMAdapter` subclass exposing only what `get_client()` needs to be exercised."""

    async def send_prompt_async(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def send_prompt_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        raise NotImplementedError


class _FakeHealthProbeClient:
    """Minimal stand-in satisfying `HealthProbeClient`'s narrow `ainvoke`/`astream` surface."""

    async def ainvoke(self, messages, **kwargs):
        raise NotImplementedError

    def astream(self, messages, **kwargs):
        raise NotImplementedError


def test_fake_health_probe_client_satisfies_the_protocol() -> None:
    """Sanity check that the test double actually structurally matches `HealthProbeClient`."""
    assert isinstance(_FakeHealthProbeClient(), HealthProbeClient)


def test_get_client_returns_the_client_attribute_set_by_a_subclass() -> None:
    """Should return whatever `self.client` a concrete adapter's __init__ set, unmodified."""
    adapter = _MinimalAdapter()
    sentinel_client = _FakeHealthProbeClient()
    adapter.client = sentinel_client

    assert adapter.get_client() is sentinel_client


def test_get_client_raises_when_no_client_attribute_is_set() -> None:
    """Should raise AdapterClientUnavailableError, naming the adapter class, instead of AttributeError.

    Covers the case `get_client()`'s docstring calls out: a future adapter that doesn't set
    `self.client` (e.g. one wrapping a non-LangChain client) must fail with a clear, distinguishable
    error rather than a generic `AttributeError` that looks like a real model outage to a caller
    like `services/health_check.py` that catches broad `Exception`s around client calls.
    """
    adapter = _MinimalAdapter()

    with pytest.raises(AdapterClientUnavailableError, match="_MinimalAdapter does not expose a 'client' attribute"):
        adapter.get_client()
