"""Adapters for external systems (providers, filesystem, etc.)."""

from __future__ import annotations

from trialagentbench_harness.adapters.llm_providers import ProviderRouting, get_provider

__all__ = ["ProviderRouting", "get_provider"]
