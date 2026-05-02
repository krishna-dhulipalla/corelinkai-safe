from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from runtime.state import PolicyGraphSettings


F = TypeVar("F", bound=Callable[..., Any])


def trace_node(name: str, fn: F, settings: PolicyGraphSettings) -> F:
    """Wrap graph nodes with LangSmith spans when tracing is enabled."""
    if not settings.langsmith_tracing:
        return fn

    try:
        from langsmith import traceable
    except ImportError:
        return fn

    return traceable(name=name, run_type="chain")(fn)  # type: ignore[return-value]
