"""Headless smoke tests for the Textual UI, using Textual's `run_test()`/Pilot API.

These run without a real terminal (`headless=True` is the default for `App.run_test()`), so they are
safe to execute from an automated agent - unlike `uv run ocht`, which launches an interactive session
that requires a human at the controls. This is a smoke test, not full UI coverage: it only verifies the
app mounts and composes without crashing after a dependency bump (e.g. the textual 3.2.0 -> 8.x upgrade).
"""
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Input

from ocht.core.db import create_db_engine, init_db
from ocht.tui.app import ChatApp


class _FailingAdapter:
    """Fake LLMAdapter that immediately raises on every call.

    Simulates e.g. a provider that is no longer reachable (connection refused).
    """

    async def send_prompt_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Raise before yielding anything, mimicking a connection error on the first chunk."""
        raise ConnectionError("Connection error.")
        yield ""  # pragma: no cover - makes this an async generator; never reached

    async def send_prompt_async(self, prompt: str, **kwargs) -> str:
        """Raise immediately, mimicking a connection error on the fallback path."""
        raise ConnectionError("Connection error.")


@pytest.fixture
def temp_db(monkeypatch):
    """Point DATABASE_URL at a fresh, empty temp SQLite file for the duration of the test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "smoke.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        init_db(create_db_engine())
        yield db_path


@pytest.mark.asyncio
async def test_app_mounts_without_crashing(temp_db):
    """Test that ChatApp composes and mounts headlessly with an empty (unconfigured) database."""
    app = ChatApp()
    async with app.run_test() as pilot:
        # No provider/model configured yet, so the app should show its onboarding message
        # rather than crash - this exercises the same startup path as a fresh install.
        assert app.query_one("#chat-input", Input) is not None
        assert app.query_one("#chat-container") is not None
        await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    ["/provider-manage", "/model-manage", "/settings", "/workspace-manage"],
)
async def test_management_screens_open_and_close(temp_db, command):
    """Test that each management screen opens (via its slash command) and closes without crashing.

    Starts from an empty DB, so on_mount() may already have pushed an onboarding modal (e.g. the
    provider selector) before the command runs - this asserts relative to that baseline rather than
    an absolute stack depth.
    """
    app = ChatApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        depth_before = len(app.screen_stack)

        await app._handle_command(command)
        await pilot.pause()
        assert len(app.screen_stack) == depth_before + 1

        app.pop_screen()
        await pilot.pause()
        assert len(app.screen_stack) == depth_before


@pytest.mark.asyncio
async def test_adapter_error_does_not_wipe_chat_container(temp_db):
    """Regression test: a failing adapter must not remove #chat-container from the DOM.

    _add_message() mounts each ChatBubble directly into #chat-container, so bot_bubble.parent IS
    the container itself. Removing `.parent` instead of the bubble (as the code briefly did) deletes
    the whole chat history area, and the very next _add_message() call then crashes with
    `NoMatches: No nodes match '#chat-container'`. This reproduces that path with a fake adapter that
    always raises, without needing a real (or now-uninstalled) LLM provider.
    """
    app = ChatApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # on_mount() already added an onboarding message since the DB is empty/unconfigured -
        # count relative to that baseline rather than an absolute number of bubbles.
        container = app.query_one("#chat-container", VerticalScroll)
        bubbles_before = len(container.children)

        app.adapter = _FailingAdapter()
        await app._process_prompt("Hi")
        await pilot.pause()

        # The container must have survived the error handling...
        container = app.query_one("#chat-container", VerticalScroll)
        assert container is not None
        # ...and it should hold the user's message plus an error bubble (the failed, empty bot
        # bubble was removed rather than the container itself).
        assert len(container.children) == bubbles_before + 2
