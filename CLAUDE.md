# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OChaT is a modular Python TUI application that orchestrates Large Language Models (LLMs) via LangChain. It supports both local models (Ollama) and cloud providers (ChatGPT, Claude, Grok).

## Development Commands

### Package Management
- `uv sync` - Install dependencies and sync environment
- `uv run ocht [command]` - Run CLI commands in development mode
- `uv run pytest` - Run tests

### Linting
- `uv run ruff check .` - Lint the codebase
- Run this after writing or modifying any Python code and fix all reported findings before
  considering the task complete. Ruleset (`[tool.ruff]` in `pyproject.toml`): pycodestyle/pyflakes
  (`E`, `F`), import sorting (`I`), bugbear (`B`), Google-style docstring checks (`D`), and
  pyupgrade (`UP` - modern syntax such as `X | None` instead of `Optional[X]`, `list[str]`
  instead of `List[str]`).
- `ruff format` is available but not enforced yet - the codebase has not been migrated to it, so
  don't run it wholesale; keep new/edited code consistent with the surrounding style by hand.
- Python target is 3.12 (`requires-python = ">=3.12"`, `.python-version`); prefer modern syntax
  (`X | None`, `list[str]`/`dict[str, str]`) over `typing.Optional`/`Union`/`List` (Ruff `UP` rules
  enforce this).

### Dependency Upgrades

- `uv audit` - Check the locked dependency set for known CVEs (uses the OSV feed; no `pip-audit`
  needed).
- The LangChain family (`langchain`, `langchain-core`, `langchain-openai`, `langchain-ollama`,
  `langchain-text-splitters`) is on the **1.x line** (`>=1.0.0,<2` in `pyproject.toml`). Migration
  history:
  - **0.3.x CVE patches (done, superseded):** an interim stage that bumped within 0.3.x to close
    every CVE with a 0.3.x backport, before the full 1.x migration below.
  - **`textual` upgrade (done):** bumped `3.2.0` -> `8.2.8` (5 majors). None of the documented
    breaking changes across those majors (`Widget.anchor` semantics, `Static`/`Label`
    `renderable` -> `content`, `HeaderTitle` reactives, Markdown component-class moves,
    `Select.BLANK` -> `Select.NULL`) touched this codebase - confirmed by grep before the bump and
    by the headless smoke tests in `tests/tui_smoke_tests.py` (uses Textual's `App.run_test()`
    Pilot API, which runs without a real terminal - safe for Claude Code to run, unlike
    `uv run ocht` itself, see Warnings section). Those smoke tests only mount the app and open/close
    each management screen against an empty DB - they are not a substitute for a human clicking
    through the real app, which still needs to happen once after any further TUI changes.
  - **LangChain 1.x migration (done):** `langchain.schema` and `langchain.memory` no longer exist
    in the `langchain` package as of 1.0. Fixes applied in `src/ocht/adapters/`:
    - Message classes (`HumanMessage`, `AIMessage`, `SystemMessage`, `BaseMessage`) and
      `BaseLanguageModel` now import from `langchain_core.messages` /
      `langchain_core.language_models` instead of the removed `langchain.schema*` paths.
    - `ConversationSummaryMemory` is gone from `langchain.memory`. Rather than pulling in
      `langchain_classic` (LangChain's official-but-deprecated compatibility package for exactly
      this class), `HybridMemoryStrategy._summarize_with_llm()` (`adapters/memory.py`) now makes
      one direct `llm.ainvoke()` call with our own summarization prompt - see that method's
      docstring for the reasoning.
    - This also fixed a real, pre-existing wiring bug found while doing the migration: each
      adapter used to keep its own *second*, outer `ConversationSummaryMemory` purely as a message
      store, but that class's `load_memory_variables()` always collapses history down to a single
      pre-summarized message (by design - it maintains one running summary string, not a message
      list). That single message was what got fed into `HybridMemoryStrategy.prepare_context()`,
      so `HybridMemoryStrategy`'s own retention/summarization logic (recent-message keeping,
      code-block priority, token trimming) never saw more than one message and was effectively
      dead code in production - only exercised directly in `tests/memory_tests.py`, never through
      the real adapter call path. The adapters (`ollama.py`, `openai_compatible.py`) now keep a
      plain `self._history: list[BaseMessage]` instead of a second memory object, so
      `HybridMemoryStrategy` receives the real, growing history. Regression coverage:
      `tests/ollama_adapter_tests.py`.
    - `use_hybrid_memory`/`memory` constructor parameters were removed from both adapters (the
      `use_hybrid_memory=False` legacy path was dead - never called anywhere in the codebase).
    - `langchain-openai`'s new "Responses API" default (which restructures `.content` into content
      blocks instead of a plain string) only activates for OpenAI's own newer endpoints - verified
      live against both Ollama and OpenRouter that `.content` stays a plain `str` with our custom
      `base_url` setups, so no `output_version="v0"` workaround was needed.
    - `langchain` 1.x now hard-depends on `langgraph` (agent orchestration framework) even though
      this project only uses the plain chat-model client - unavoidable extra weight from staying
      on `langchain`, unrelated to anything we do with it.
    - `uv audit` confirms all previously-open LangChain-family CVEs are closed at the 1.x
      versions now resolved, with no new CVEs introduced by `langgraph`/`langgraph-checkpoint`/
      `langgraph-prebuilt`/`langgraph-sdk`/`langchain-protocol`/`ormsgpack`.
  - Transitive dependencies with CVEs but no direct import in this codebase (`langsmith`, `mako`,
    `pygments`, `requests`, `urllib3`) are floored via `[tool.uv].constraint-dependencies` instead
    of being added to `dependencies`, since we don't call their APIs directly.

### Provider Routing Preferences (`LLMProviderConfig.prov_params`)

- `prov_params` is a JSON object of OpenAI-compatible "extra body" routing preferences, forwarded
  verbatim to the provider via `ChatOpenAI(extra_body={"provider": <prov_params>})`
  (`_build_openai_compatible_params()` in `services/adapter_manager.py`). Only wired for the
  `openai`/`lm studio`/`openrouter` branch (`OpenAICompatibleAdapter`) - not for Ollama, whose
  equivalent tuning knobs (`options`) have different semantics and aren't wired up.
- For OpenRouter specifically, this is their [provider routing
  object](https://openrouter.ai/docs/guides/routing/provider-selection) - e.g.
  `quantizations` (list of acceptable quantization levels) and `preferred_min_throughput`
  (tokens/sec; a soft preference that deprioritizes slower endpoints rather than excluding them -
  unlike `quantizations`, which hard-excludes and can 404 a request if no endpoint matches).
- The OpenRouter provider row in this project's DB is currently set to
  `{"quantizations": ["int8", "fp8", "mxfp8", "fp16", "bf16", "fp32", "unknown"],
  "preferred_min_throughput": 40}` - i.e. "q8-or-better, or undisclosed quantization." `unknown`
  is deliberately included: 3 of the 7 configured OpenRouter models (`qwen/qwen3.8-flash`,
  `qwen/qwen3.8-max`, `deepseek/deepseek-v4-flash-vision-exp`) only have endpoints with undisclosed
  quantization today, so excluding `unknown` would 404 every request for those models. Check
  `GET /api/v1/models/{model}/endpoints` before tightening this filter.
- A malformed `prov_params` value is ignored gracefully (falls back to default provider routing)
  rather than blocking the adapter switch - see `_build_openai_compatible_params()`.
- Editable via the UI: `tui/screens/provider_manager.py`'s provider edit form has a "Routing
  Params" field. Validated as JSON before saving (`services/provider_manager.py`'s
  `_validate_provider_params()`); clearing the field explicitly clears `prov_params` (empty string
  is unambiguous - a JSON string can never legitimately be empty - see
  `update_llm_provider_config()`'s docstring for the `None` vs `""` distinction).

### Model-Level Generation Parameters (`Model.model_params`)

- `model_params` is a JSON object of default generation parameters for one specific model (e.g.
  `{"temperature": 1.0, "num_predict": 32768}`), forwarded verbatim into the LangChain chat-model
  constructor (`ChatOllama`/`ChatOpenAI`) via `_merge_model_params()` in
  `services/adapter_manager.py`. It is merged on top of the provider-level `default_params`
  (the hardcoded `{"temperature": 0.5}` for Ollama, or `_build_openai_compatible_params()`'s
  `{"temperature": 0.7, ...}` for the OpenAI-compatible branch) - model-level keys win on
  conflicts, since a setting scoped to one model is more specific than a provider-wide default.
- This closes a real gap: the `model_params` column existed in the schema and was already editable
  via `tui/screens/model_manager.py` before this, but `_create_adapter()` never read it back - the
  hardcoded per-provider temperature was applied unconditionally regardless of which model was
  selected. Generation parameters like temperature/max_tokens are a property of the model being
  queried, not of the transport used to reach it, so they belong on `Model` rather than on
  `LLMProviderConfig` or in a new provider-scoped config object.
- A malformed or non-dict `model_params` value is ignored gracefully (falls back to the
  provider-level params alone) rather than blocking the adapter switch - same pattern as
  `prov_params` above.
- `max_tokens` and `context_window` (Ollama-only) now have dedicated, validated fields in the
  model edit form (`tui/screens/model_manager.py`) instead of requiring the raw key names to be
  hand-typed into the JSON blob - see `services/model_manager.py`'s `build_model_params_json()`/
  `split_model_params_for_editing()`. `streaming_enabled` was dropped from consideration:
  `tui/app.py:_process_prompt()` already always attempts streaming first and falls back
  automatically, so a manual per-model toggle would solve a problem that doesn't exist.

#### Current Ollama `model_params` values and their sourcing

The four locally-installed Ollama models each have a `model_params` row, set with values sourced
from two places rather than guessed:

- **Sampling params (`temperature`, `top_p`, `top_k`) for the three `qwen3.8:27b-*` variants**
  (`-mxfp8`, `-mlx`, `-nvfp4` - same underlying `qwen3_5` weights, different quantizations) were
  copied from this machine's working OpenCode config (`~/.config/opencode/opencode.json`, its
  "medium reasoning effort" profile for the same models): `{"temperature": 1.0, "top_p": 0.95,
  "top_k": 20}`. This is deliberately **not** a low temperature - these are reasoning/"thinking"
  models (`ollama show` lists the `thinking` capability), and lowering temperature on a
  reasoning-model's chain-of-thought is known to degrade it (the same failure mode documented for
  Qwen3 and DeepSeek-R1) rather than making code output more precise the way it would for a
  non-reasoning model.
- **`gemma4:31b-mxfp8`** has no override in that OpenCode config at all, so its sampling params
  were left at Ollama's own native defaults (`{"temperature": 1.0, "top_p": 0.95, "top_k": 64}`
  per `ollama show gemma4:31b-mxfp8`) rather than invented.
- **`num_ctx: 262144`** (each model's native max, from `ollama show`) and **`num_predict: 32768`**
  are the same for all four models and are *not* from OpenCode (its config doesn't set either).
  They were chosen after empirically checking actual memory use via `ollama ps` on this machine's
  Mac Studio M4 (64 GB unified memory): loading `qwen3.8:27b-mxfp8` or `gemma4:31b-mxfp8` with the
  full native `num_ctx` reported only ~32 GB total, 100% GPU-resident - comfortable headroom, so no
  need to trade context length down for memory safety on this hardware. `num_predict: 32768` is a
  deliberate cap (not `-1`/unbounded) to allow generating very large files/functions while still
  guarding against runaway generations, sized relative to the now-large context budget.
- **Known gap:** OpenCode's Qwen profile also sets `min_p: 0.0` and `presence_penalty: 0.0`.
  `langchain-ollama`'s `ChatOllama` (the version pinned in this project) has no constructor field
  for either - confirmed via `ChatOllama.model_fields` and its `_chat_params()` source, which
  builds the request's `options` dict from a fixed, hardcoded list of fields that does not include
  `min_p` or `presence_penalty`. There is currently no way to set them through this adapter; a
  `model_params` JSON key for either would simply be dropped, not forwarded.

### Memory Configuration via App Settings (`MemoryConfig`)

- Every `HybridMemoryStrategy` tunable (`adapters/memory.py`'s `MemoryConfig` dataclass -
  `max_context_tokens`, `recent_messages_count`, `code_retention_priority`,
  `summarization_threshold`) used to be an unconditional hardcoded default: no adapter anywhere
  ever constructed a `MemoryConfig` with custom values, so every conversation ran with
  `max_context_tokens=4000` etc. regardless of provider/model/workflow - the same "field exists
  but nothing ever overrides it" gap `Model.model_params` had before "Model-Level Generation
  Parameters" above fixed it.
- `services/adapter_manager.py`'s `_build_memory_config()` now reads each field from the generic
  `Setting` key-value store instead, via well-known keys on `AdapterManager`:
  `MEMORY_MAX_CONTEXT_TOKENS_KEY` ("memory_max_context_tokens"),
  `MEMORY_RECENT_MESSAGES_COUNT_KEY` ("memory_recent_messages_count"),
  `MEMORY_CODE_RETENTION_PRIORITY_KEY` ("memory_code_retention_priority"),
  `MEMORY_SUMMARIZATION_THRESHOLD_KEY` ("memory_summarization_threshold"). `_create_adapter()`
  calls it once per adapter switch and passes the result as `memory_config=` to both
  `OllamaAdapter`/`OpenAICompatibleAdapter` (also previously never wired - both adapters always
  fell back to their own internal `MemoryConfig()` default).
- These are global settings, not per-model ones (unlike `Model.model_params`): the memory
  strategy governs how conversation history is trimmed/summarized regardless of which
  provider/model is currently active, so one `Setting` row applies across all of them.
- Editable today via the existing generic Settings Manager screen
  (`tui/screens/settings_manager.py`, Ctrl+N in the Settings screen) - add a row with one of the
  four key names above and an integer (or float, for `code_retention_priority`) value. No
  dedicated form/labels exist for these yet, unlike the typed `max_tokens`/`context_window` model
  fields; it's the same raw key-value editing the screen already offered for
  `current_provider_id`/`current_model_name`.
- A missing, non-numeric, or non-positive value for any of the four keys falls back to that
  field's original hardcoded default (`_read_positive_int_setting`/`_read_positive_float_setting`)
  rather than blocking adapter creation - same "ignore malformed config" precedent as
  `prov_params`/`model_params`.

### Health Check System & Resilience Layer

- **Health checks** (`services/health_check.py`) send one real completion request per model,
  unlike `sync_llm_models()`'s existing sync flow, which only confirms a model is *listed* by its
  provider (`GET /api/tags` for Ollama, `GET /models` for LM Studio) and never validates that it
  can actually complete a request. Tries the streaming path first (`_try_streaming()`), falls back
  to non-streaming (`_try_non_streaming()`) on failure, and records which path succeeded, latency,
  tokens/second, and any error onto the `Model` row via `record_health_check_result()`
  (`Model.last_check_latency_ms`/`last_check_tokens_per_second`/`last_check_error`, alongside the
  pre-existing `is_available`/`last_checked` - both the old listing-based sync and the new
  completion-based check share those two columns rather than each having their own conflicting
  availability flag).
- **`HEALTH_CHECK_PROMPT`** asks for a ~200-word short story, not a trivial one-token reply like
  "Say OK." - tokens/second measured off a 1-3 token response is dominated by connection/prompt-eval
  overhead rather than real sustained throughput (observed swinging ~19-50 tok/s run to run on the
  same model with the old one-token prompt); a longer generation gives a materially more stable
  reading (~28-30 tok/s across repeated runs in practice) at the cost of a slightly slower, slightly
  more expensive (for paid providers) check.
- **Tokens/second** (`_tokens_per_second()`) prefers Ollama's native `eval_count`/`eval_duration`
  from `response_metadata` (pure generation time, excludes prompt-eval/network overhead) and falls
  back to `usage_metadata['output_tokens']` divided by client-measured wall-clock time for every
  other provider (OpenAI-compatible responses carry no timing metadata at all).
- **Triggered on-demand only** - a "💓 Health Check" button/Ctrl+H binding in
  `tui/screens/model_manager.py`, and `ocht health-check [--provider-id ID] [--model NAME]` on the
  CLI (checks one model, one provider's models, or every model across every provider - including
  paid cloud ones - depending on which flags are given). No scheduler/background timer exists in
  this codebase, and none was introduced for this - a deliberate v1 scope decision.
- **Never touches the active chat adapter**: `services/adapter_manager.py`'s `build_adapter(db,
  provider_config, model)` was extracted from `AdapterManager._create_adapter()` specifically so
  health checks can build a throwaway adapter for any (provider, model) pair without mutating
  `AdapterManager`'s `_current_adapter`/`_current_provider_id`/`_current_model_name` singleton -
  checking model B's health must never disrupt whichever model the user is actively chatting with.
- **Bypasses the retry/circuit-breaker layer below on purpose**: a health check's entire point is
  to report the model's *current, true* status - masking a failure behind automatic retries would
  produce a falsely-rosy result. It talks to `adapter.client` (the raw LangChain client each
  concrete adapter exposes by convention, not a declared part of `LLMAdapter`'s interface)
  directly, bypassing `send_prompt_async`/`send_prompt_stream` entirely. **Known limitation**: this
  convention isn't enforced by the type system - a future adapter that doesn't wrap a LangChain
  client (e.g. the Phase 3 MLX-LM adapter) would need either a `client`-compatible shim or a
  formalized contract on `LLMAdapter` (a review flagged this as worth addressing when that adapter
  is built, not before - `check_model_health()` already reports a clear, distinguishable error
  ("... does not expose a 'client' attribute...") rather than a confusing `AttributeError` in that
  case, so it fails legibly today even without the formalized contract).

- **Error recovery & retry** (`adapters/resilience.py`) wraps the adapters' actual LangChain
  client calls (`self.client.ainvoke()`/`self.client.astream()`) with exponential backoff
  (`RetryPolicy`) and a circuit breaker (`CircuitBreaker`), via `LLMAdapter.__init__()`'s new
  `_resilient_ainvoke()`/`_resilient_astream()` helpers - `OllamaAdapter`/`OpenAICompatibleAdapter`
  route through these instead of calling `self.client` directly, with no change to their public
  `send_prompt_async`/`send_prompt_stream` signatures.
- **Exception classification** (`is_retryable()`): rate limits (429) and connection/timeout/5xx
  errors are retried; authentication/permission/not-found/bad-request errors (and any unrecognized
  exception type) fail immediately rather than being masked as transient.
- **Streaming retries only before the first chunk arrives** (`stream_with_resilience()`) -
  retrying transparently after a chunk has already been yielded (and, in the TUI, already rendered
  into a chat bubble) would duplicate or reset visible content, so a post-first-chunk failure is
  recorded on the circuit breaker but re-raised immediately, letting `tui/app.py`'s pre-existing
  stream→async fallback (`_process_prompt`/`_process_prompt_fallback`, unchanged) handle it exactly
  as before. The two layers are complementary: retry/circuit-breaker is a *resilience* strategy
  (retry the same call shape), the TUI's fallback is a *degradation* strategy (switch shape) -
  a stream that exhausts its retries still falls through to the existing fallback.
- **Circuit breaker state is per-adapter-instance**, not persisted across provider/model switches
  or app restarts - it resets naturally every time `build_adapter()` constructs a fresh adapter.
  Documented v1 limitation, not an oversight; note that this also means re-`switch_adapter()`-ing
  to the *same* (provider, model) after a failure gets a fresh, closed breaker rather than an
  accumulating failure count.
- `httpx`/`openai` were promoted from transitive (via `langchain-openai`/`ollama`) to explicit
  `pyproject.toml` dependencies, since `resilience.py` imports them directly for exception
  classification (`uv audit` clean at these versions).

### Anthropic Adapter (`adapters/anthropic.py`)

- `AnthropicAdapter` follows the exact `OpenAICompatibleAdapter` template: same constructor shape
  (`model`, `api_key`, `base_url`, `default_params`, `memory_config`, `retry_policy`), same
  `send_prompt_async`/`send_prompt_stream` bodies routed through `_resilient_ainvoke`/
  `_resilient_astream`, same `self.client`/`self.memory_strategy`/`self._history` attributes.
- Wired in `build_adapter()` (`services/adapter_manager.py`) as its own top-level branch
  (`prov_name == "anthropic"`, case-insensitive) rather than folded into the OpenAI-compatible
  branch, since `ChatAnthropic` is a distinct LangChain client class (Anthropic's Messages API,
  not the OpenAI chat-completions shape) even though the adapter code around it looks nearly
  identical.
- `prov_api_key` is always passed through explicitly from the DB row - `ChatAnthropic`'s own
  `ANTHROPIC_API_KEY` env-var auto-detection is never consulted, same precedent as every other
  adapter (no adapter here ever relies on ambient provider SDK env vars).
- `ChatAnthropic`'s own `max_retries` client-level retry knob (default `2`) is set to `0` in the
  adapter's constructor - retries are owned exclusively by `adapters/resilience.py`'s
  `RetryPolicy`/`CircuitBreaker`, so a transient failure isn't silently retried twice (once inside
  the `anthropic` SDK, once by this project's own resilience layer) with no visibility/consistent
  backoff for the second layer. Applied *after* `default_params` is merged into the constructor's
  `client_kwargs` dict (an architecture review caught this: an earlier version applied it before
  the merge, so a `max_retries` key inside a model's free-form `model_params` JSON could silently
  win and re-enable client-level retries - `tests/anthropic_adapter_tests.py`'s
  `test_default_params_cannot_override_max_retries` locks in the fix).
- `adapters/resilience.py`'s `is_retryable()`/`compute_delay()` were extended with an
  `anthropic.*`-specific branch (`RateLimitError`/`APIConnectionError`/`APITimeoutError`
  retryable, `AuthenticationError`/`NotFoundError`/`PermissionDeniedError`/`BadRequestError`/
  `ConflictError`/`UnprocessableEntityError` not) - the `anthropic` SDK's exception hierarchy is
  structurally similar to `openai`'s but is a distinct set of classes, so this could not simply
  reuse the existing OpenAI branch. Verified live against the installed `anthropic==0.125.0`.
- No OpenRouter-style `extra_body`/provider-routing concept exists for Anthropic, so
  `_build_anthropic_params()` (`services/adapter_manager.py`) does not read `prov_params` at all -
  only `Model.model_params` (via the existing `_merge_model_params()`) is available for per-model
  Anthropic tuning (`temperature`, `max_tokens`, `top_p`, `top_k`). No `max_tokens` default is
  hardcoded either - `langchain-anthropic` (confirmed against the installed `1.7.0`) already
  defaults it to a large, sensible value when unset.
- No auto-sync (`_sync_anthropic_models`) - follows the existing OpenAI/OpenRouter manual-entry
  precedent for cloud APIs (only self-hosted providers get `sync_llm_models()` auto-sync).
- Health check (`services/health_check.py`) needed zero changes: `usage_metadata`'s
  `input_tokens`/`output_tokens` shape is identical to every other provider's, and Anthropic's
  `response_metadata` carries no timing field analogous to Ollama's `eval_duration` - so
  `_tokens_per_second()` always takes its existing wall-clock-fallback branch for Anthropic, the
  same branch already used for OpenAI-compatible responses.
- Third adapter to duplicate `_convert_tuples_to_messages()` motivated finally extracting it into
  `LLMAdapter` (`adapters/base.py`) as a shared concrete method - `ollama.py`/`openai_compatible.py`
  had it as byte-for-byte identical private copies before this.
- **Known limitation, flagged by architecture review**: `ChatAnthropic` raises a `ValueError` if
  the message list passed to it contains more than one *non-consecutive* `SystemMessage` -
  a constraint `ChatOllama`/`ChatOpenAI` don't have. Today this is safe only because
  `HybridMemoryStrategy.prepare_context()` never emits more than one `"system"`-role tuple per
  call (the "Previous conversation summary" line) - documented on
  `LLMAdapter._convert_tuples_to_messages()`, but not enforced anywhere. If `prepare_context()` is
  ever extended to emit a second system-tagged tuple, it must be verified against a real
  (unmocked) `AnthropicAdapter` call - the existing tests mock `ChatAnthropic` entirely and would
  not catch this.
- **Tracked technical debt, not addressed here**: `adapters/resilience.py`'s `is_retryable()`/
  `compute_delay()` now repeat a near-identical four-branch pattern per SDK (OpenAI, Anthropic,
  plus Ollama's own shape) - an Open/Closed violation that will keep growing linearly with each
  new adapter's SDK. Worth refactoring into a small data-driven table (one `SdkExceptionProfile`
  per SDK, looped over once) before a fourth adapter's exception hierarchy is added - deferred
  here since it's a pure refactor with no behavior change, not blocking this feature.

### MLX-LM: Provider Recognition, Not a Native Adapter

- MLX-LM support does **not** get its own adapter class. `mlx_lm.server` (Apple's own `mlx-lm`
  PyPI package, actively maintained by the ml-explore org) ships a local, OpenAI-compatible HTTP
  server (`/v1/chat/completions` with real SSE streaming, `/v1/models`) - exactly the same shape
  LM Studio and OpenRouter already speak - so MLX-LM is wired as a new recognized `prov_name`
  (`"mlx-lm"`, hyphen required) in `build_adapter()`'s existing `OpenAICompatibleAdapter` branch,
  alongside `"openai"`/`"lm studio"`/`"openrouter"`.
- This was a deliberate decision, not an oversight: the only existing in-process LangChain
  integration for MLX (`langchain_community.chat_models.mlx.ChatMLX` +
  `langchain_community.llms.mlx_pipeline.MLXPipeline`) is a dead end - `langchain-community` was
  archived/sunset by LangChain in June 2026, a maintainer-proposed standalone `langchain-mlx`
  package was explicitly closed as "not planned," `ChatMLX` has no async streaming (`_agenerate()`
  exists but no `_astream()`), populates no `usage_metadata`/`response_metadata` at all (would
  break `health_check.py`'s generic token/timing extraction entirely), and has an unresolved
  broken-tool-calling bug (`langchain-community#308`). A bespoke adapter built directly against
  `mlx_lm.load()`/`stream_generate()` would mean hand-rolling token counting and a
  LangChain-compatible response shape from scratch, with no ecosystem precedent to build on.
- Net effect of routing through `OpenAICompatibleAdapter` instead: real async streaming, real
  `usage_metadata` (so `health_check.py` needed zero changes), and no new adapter file at all.
- The user runs `mlx_lm.server` themselves as a separate local process (`python -m mlx_lm.server
  --model <path-or-hf-repo>`), exactly like `ollama serve`/the LM Studio app already are - it is
  not a Python dependency of OChaT, since OChaT only ever talks HTTP to it. No new
  `pyproject.toml` dependency was needed for MLX-LM support.
- `prov_api_key` needs no real credential - by convention (mirroring the existing Ollama row,
  which stores `"0"`), enter a placeholder value since the field is non-nullable but
  `mlx_lm.server` doesn't validate any `Authorization` header by default.
- **Scope decision**: no auto-sync (`_sync_mlxlm_models`) - models are added manually via the
  Model Manager TUI, matching the OpenAI/OpenRouter precedent rather than the Ollama/LM Studio
  one, to keep this addition focused (the sync-function family also has no existing test coverage
  to build on today).

### Building and Distribution
- `uv build` - Build the package using setuptools
- `uv install -e .` - Install package in editable mode

### Running the Application
- `uv run ocht` - Launch default chat interface
- `uv run ocht init <workspace>` - Create new workspace
- `uv run ocht chat` - Start interactive chat
- `uv run ocht list-models` - List available models
- `uv run ocht sync-models` - Sync model metadata
- `uv run ocht migrate <version>` - Run database migrations

## Architecture Overview

### Core Components

**Database Layer (`core/`)**
- `models.py` - SQLModel entities: Workspace, Message, LLMProviderConfig, Model, Setting, PromptTemplate
- `db.py` - Database engine, session management, and initialization
- `migration.py` - Alembic integration for schema migrations

**Repository Layer (`repositories/`)**
- CRUD operations for each entity
- Direct database access abstraction
- Files: `workspace.py`, `message.py`, `llm_provider_config.py`, `model.py`, `setting.py`, `prompt_template.py`

**Service Layer (`services/`)**
- Business logic and use cases
- Orchestrates repositories and external APIs
- Files: `workspace.py`, `chat.py`, `config.py`, `model_manager.py`, `provider_manager.py`, `prompt_manager.py`

**Adapter Layer (`adapters/`)**
- LangChain integration
- `base.py` - Abstract LLMAdapter interface
- `ollama.py` - Ollama-specific implementation

**TUI Layer (`tui/`)**
- Textual-based user interface
- `app.py` - Main TUI application
- `screens/` - UI screens for model/provider management
- `widgets/` - Custom UI components (chat bubbles, etc.)
- `styles/` - TCSS styling files

### Data Models

**Primary Entities:**
- `Workspace` - Chat workspace container
- `Message` - Individual chat messages with role, content, metadata
- `LLMProviderConfig` - API credentials and provider settings
- `Model` - Available LLM models per provider
- `Setting` - Key-value configuration storage
- `PromptTemplate` - Reusable prompt templates

**Key Relationships:**
- Workspace → Messages (1:many)
- LLMProviderConfig → Models (1:many)
- Workspace → default_model (foreign key to LLMProviderConfig)
- Messages support threading via parent_id self-reference

### CLI Structure

CLI commands map to service layer functions:
- `init` → `workspace.create_workspace()`
- `chat` → `chat.start_chat()`
- `config` → `config.open_conf()`
- `list-models` → `model_manager.list_llm_models()`
- `sync-models` → `model_manager.sync_llm_models()`

### Database Configuration

- Default: SQLite at `data/ocht.db`
- Configurable via `DATABASE_URL` environment variable
- Uses SQLModel/SQLAlchemy for ORM
- Alembic for schema migrations

## Development Notes

### Database Initialization
The database is automatically initialized when first accessed. Use `init_db()` to create tables manually.

### Adding New Providers
1. If the provider speaks the OpenAI chat-completions API (like MLX-LM does), just add its
   `prov_name` string to `build_adapter()`'s existing `OpenAICompatibleAdapter` branch
   (`services/adapter_manager.py`) - no new adapter class needed. Otherwise, create a new adapter
   in `adapters/` extending `LLMAdapter` (see `adapters/anthropic.py` for the current template).
2. Wire it into `build_adapter()` (`services/adapter_manager.py`) - this is the actual dispatch
   point (keyed on `provider_config.prov_name.lower()`), not `provider_manager.py` (which only
   handles provider-row CRUD/validation - `prov_name` is free text with no enum anywhere).
3. `LLMProviderConfig`'s existing fields (`prov_api_key`, `prov_endpoint`, `prov_params`) already
   cover most connection needs; for a provider with no real API key concept, store a placeholder
   value (e.g. `"0"`, the existing Ollama convention) rather than relaxing the non-nullable field.
4. If the new adapter should raise its own SDK-specific exceptions, add an `is_retryable()` branch
   in `adapters/resilience.py` for them - don't assume an existing provider's exception hierarchy
   covers a new one (verified needed for Anthropic; `anthropic.*` exceptions are a distinct set of
   classes from `openai.*`, not a shared base type, despite similar shapes).
5. Add TUI screens if needed (usually not - the Provider/Model Manager screens are already
   provider-agnostic).

### Testing
Tests are configured via pytest. Use `uv run pytest` to run the test suite.

### Workspace Management
Workspaces are self-contained chat environments with their own configuration and message history. Each workspace references a default model configuration.

## Warnings and Precautions
- Starte nie die App mit uv run ocht da es sich um eine TUI App handelt die du nicht steuern kannst!

## Development Guidelines
- Merke Debug in der TUI Anwendung nur mit self.notify() erstellen

## Translation Guidelines
- Alle Texte in der Anwendung in englisch erstellen.

## Adapter Roadmap & Next Steps

### Phase 1: Memory System Improvements (High Priority)
- [x] Implement HybridMemoryStrategy
  - [x] Keep last 8-10 messages completely (for code context) - `MemoryConfig.recent_messages_count`
  - [x] Smart summarization for older messages - `HybridMemoryStrategy._summarize_with_llm()`; only
    actually reachable in production since the LangChain 1.x migration fixed a wiring bug where
    adapters fed it a single pre-summarized message instead of real history (see "Dependency
    Upgrades" above) - before that, this logic was dead code despite the class existing
  - [x] Code blocks retained longer than natural text - `_select_important_messages()`
  - [x] Token-aware context management - `_trim_to_token_limit()`

### Phase 2: Configuration & Health Monitoring (Medium Priority)
- [x] Model-scoped generation config (formerly framed as a "provider-agnostic AdapterConfig
      class" - reframed because temperature/max_tokens/context_window are properties of the model
      being queried, not of the provider/transport used to reach it; see "Model-Level Generation
      Parameters" above)
  - [x] temperature: per-model override flows through `Model.model_params` ->
    `_merge_model_params()`, taking precedence over the provider-level hardcoded default
    (`{"temperature": 0.5}` for Ollama, `0.7` for OpenAI-compatible)
  - [x] max_tokens: dedicated, validated "Max Tokens" field in `tui/screens/model_manager.py`'s
    model edit form. `services/model_manager.py`'s `build_model_params_json()`/
    `split_model_params_for_editing()` translate the one generic UI field to/from the key each
    adapter actually expects (`num_predict` for Ollama, `max_tokens` for OpenAI-compatible) and
    merge it with the free-form "Advanced Params" field (temperature/top_p/top_k/...) into the
    same single `model_params` JSON column
  - [x] context_window: dedicated "Context Window" field, Ollama-only (`num_ctx`) - disabled in
    the UI for any other provider, since no other supported adapter's LangChain client exposes an
    equivalent constructor parameter (context length is a fixed property of the hosted model on
    those providers, not a request-time setting)
  - [~] streaming_enabled: dropped from this list rather than built - `tui/app.py:_process_prompt()`
    already always attempts `send_prompt_stream()` first and falls back to the async method on
    failure, so a manual per-model toggle would be a solution without an observed problem;
    revisit only if a concrete case surfaces where that automatic fallback isn't good enough
  - [x] memory_strategy tuning: `MemoryConfig` is now actually constructed with real values (via
    `_build_memory_config()`, see "Memory Configuration via App Settings" above) and passed
    through to both adapters, instead of always silently falling back to hardcoded defaults;
    swapping in a different `MemoryStrategy` *implementation* is still only architecturally
    possible - only `HybridMemoryStrategy` exists (see Phase 4)
  - `LLMProviderConfig.prov_params` (see "Provider Routing Preferences" above) remains the
    provider-scoped counterpart: OpenRouter-specific provider-routing prefs, not model parameters
- [x] Health check system - see "Health Check System & Resilience Layer" below
  - [x] Model availability testing - real completion call via `services/health_check.py`, not just
    `sync_llm_models()`'s provider-listing probe
  - [x] Response time monitoring - `Model.last_check_latency_ms`
  - [x] Streaming capability validation - tries the streaming path first, falls back to
    non-streaming, records which path (`"stream"`/`"async"`) succeeded
  - Bonus (not originally scoped, requested during implementation): tokens/second -
    `Model.last_check_tokens_per_second`
- [x] Error recovery & retry logic - see "Health Check System & Resilience Layer" below
  - [x] Exponential backoff - `adapters/resilience.py`'s `RetryPolicy`/`compute_delay()`
  - [x] Circuit breaker pattern - `adapters/resilience.py`'s `CircuitBreaker`, one per adapter
    instance
  - [x] Graceful degradation (stream → async → error): the pre-existing `_process_prompt` ->
    `_process_prompt_fallback()` TUI-layer fallback is unchanged, but now composes with the new
    adapter-layer retry/circuit-breaker (see below) instead of being the only resilience mechanism

### Phase 3: New Adapters (Medium Priority)
- [x] OpenAI-compatible adapter (OpenAI, Groq, local APIs) - `OpenAICompatibleAdapter`; also now
  covers OpenRouter (see "Provider Routing Preferences" above) and MLX-LM (see "MLX-LM: Provider
  Recognition, Not a Native Adapter" below)
- [x] MLX-LM adapter (Apple Silicon local models) - not a native adapter class, see above
- [x] Anthropic Claude adapter (API) - `AnthropicAdapter`, see "Anthropic Adapter" below

### Phase 4: Advanced Features (Low Priority)
- [ ] Context-aware parameter adjustment
- [ ] Multi-model conversation support
- [ ] Adapter performance metrics
- [ ] Custom memory strategies per use case - only `HybridMemoryStrategy` exists; `MemoryStrategy`
  is an ABC so this is architecturally possible, just nothing else implements it yet

### Current Status (2026-08-28)
✅ Base LLMAdapter with async/sync/stream methods
✅ OllamaAdapter and OpenAICompatibleAdapter (OpenAI/LM Studio/OpenRouter) with streaming support
✅ TUI streaming implementation with live updates
✅ Mouse escape sequence filtering
✅ HybridMemoryStrategy actually active in production (see Phase 1)
✅ Per-model temperature override via `Model.model_params` -> `_merge_model_params()` (see
   "Model-Level Generation Parameters" and Phase 2)
✅ Dedicated "Max Tokens"/"Context Window" fields in the model edit form, backed by the same
   `model_params` column (Phase 2 - see `build_model_params_json()`/`split_model_params_for_editing()`)
✅ Health Check System (real completion-based checks, tokens/sec, on-demand via TUI/CLI) and
   Error Recovery/Retry Logic (exponential backoff + circuit breaker at the adapter layer) - see
   "Health Check System & Resilience Layer" (Phase 2, both items closed)
✅ AnthropicAdapter and MLX-LM provider recognition via `OpenAICompatibleAdapter` (Phase 3, both
   items closed) - see "Anthropic Adapter" / "MLX-LM: Provider Recognition, Not a Native Adapter"
🟡 Next most natural steps: formalizing the `adapter.client` convention `health_check.py` relies
   on (see that section's "Known limitation" - now three adapters share this informal contract,
   not two), or starting Phase 4's advanced features