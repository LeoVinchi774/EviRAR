"""Session-scoped OpenAI-compatible clients and request accounting."""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from threading import BoundedSemaphore, Lock
from typing import Any, Callable

from openai import OpenAI

from evirar.config import Settings

_CURRENT: ContextVar["Runtime"] = ContextVar("evirar_runtime")


def current_runtime() -> "Runtime":
    try:
        return _CURRENT.get()
    except LookupError as exc:
        raise RuntimeError("Use Consultation or Runtime.activate() before calling core components") from exc


def bind_runtime(function: Callable) -> Callable:
    """Propagate session settings into validation worker threads."""
    context = copy_context()

    def wrapped(*args, **kwargs):
        return context.copy().run(function, *args, **kwargs)
    return wrapped


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.events: list[dict[str, Any]] = []
        self._clients: dict[str, OpenAI] = {}
        self._lock = Lock()
        self.validation_semaphore = BoundedSemaphore(
            settings.values["retrieval"]["validation_workers"]
        )

    @contextmanager
    def activate(self):
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def client(self, kind: str) -> OpenAI:
        with self._lock:
            if kind not in self._clients:
                endpoint = getattr(self.settings, kind)
                endpoint.validate(kind)
                options = self.settings.values["runtime"]
                self._clients[kind] = OpenAI(
                    api_key=endpoint.api_key, base_url=endpoint.base_url,
                    timeout=options["timeout_seconds"], max_retries=options["max_retries"],
                )
            return self._clients[kind]

    def _record(self, service: str, label: str, start: float, response=None, error: Exception | None = None):
        usage = getattr(response, "usage", None)
        event = {"service": service, "label": label, "elapsed_seconds": time.perf_counter() - start,
                 "ok": error is None, "usage": usage.model_dump() if usage is not None else None}
        if error is not None:
            event["error_type"] = type(error).__name__
        with self._lock:
            self.events.append(event)

    def chat(self, messages: list[dict[str, str]], *, service: str = "llm", label: str = "doctor",
             temperature: float | None = None, max_tokens: int | None = None,
             response_format: dict | None = None, **overrides) -> str:
        # Simulator/evaluator requests must not inherit the doctor's repetition
        # penalty or vLLM-specific sampling options.
        generation = self.settings.values["simulator_generation" if service == "simulator" else "generation"]
        args = {key: generation[key] for key in ("temperature", "max_tokens", "top_p", "presence_penalty")}
        if service == "simulator" or label in {"doctor", "doctor_forced_diagnosis"}:
            args["stop"] = ["<|im_end|>"]
        if temperature is not None:
            args["temperature"] = temperature
        if max_tokens is not None:
            args["max_tokens"] = max_tokens
        args.update(overrides)
        if generation["extra_body"]:
            args["extra_body"] = generation["extra_body"]
        if response_format is not None:
            args["response_format"] = response_format
        start = time.perf_counter()
        try:
            response = self.client(service).chat.completions.create(
                model=getattr(self.settings, service).model, messages=messages, **args,
            )
            text = response.choices[0].message.content
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Chat service returned an empty text response")
            self._record(service, label, start, response)
            return text
        except Exception as exc:
            self._record(service, label, start, error=exc)
            raise

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_size = self.settings.values["runtime"]["embedding_batch_size"]
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset + batch_size]
            start = time.perf_counter()
            try:
                response = self.client("embedding").embeddings.create(
                    model=self.settings.embedding.model, input=batch,
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in ordered] != list(range(len(batch))):
                    raise ValueError("Embedding response indexes do not match the input batch")
                for item in ordered:
                    vector = item.embedding
                    if not vector or not all(math.isfinite(v) for v in vector) or not any(vector):
                        raise ValueError("Embedding service returned an invalid vector")
                    if vectors and len(vector) != len(vectors[0]):
                        raise ValueError("Embedding dimensions changed within a request")
                    vectors.append(vector)
                self._record("embedding", "embedding", start, response)
            except Exception as exc:
                self._record("embedding", "embedding", start, error=exc)
                raise
        return vectors

    def ensure_success(self, since: int = 0):
        failures = [event for event in self.events[since:] if not event["ok"]]
        if failures:
            raise RuntimeError(f"{len(failures)} API calls failed; refusing to report a complete EviRAR result")

    def close(self):
        for client in self._clients.values():
            client.close()

