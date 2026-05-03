"""Provider error classification and noisy retry-log filtering."""

from __future__ import annotations

import logging
from typing import Any


RATE_LIMIT_SPAM = "OpenAI threw rate limit error"


class ProviderRetryNoiseFilter(logging.Filter):
    """Suppress repeated provider retry messages while preserving final errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        return RATE_LIMIT_SPAM not in record.getMessage()


def install_provider_retry_noise_filter() -> None:
    logging.getLogger().addFilter(ProviderRetryNoiseFilter())
    logging.getLogger("strands").addFilter(ProviderRetryNoiseFilter())
    logging.getLogger("openai").addFilter(ProviderRetryNoiseFilter())


def classify_provider_error(error: Any) -> dict[str, str]:
    text = str(error or "")
    lowered = text.lower()
    if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
        return {
            "kind": "quota_or_billing",
            "message": "Provider quota/billing is exhausted. This is different from requests-per-minute throttling.",
            "detail": text,
        }
    if "rate_limit" in lowered or "rate limit" in lowered or " too many requests" in lowered or "429" in lowered:
        return {
            "kind": "rate_limit",
            "message": "Provider rate limit hit. This usually means too many requests/tokens too quickly; retry after a delay or use --agent-delay-seconds.",
            "detail": text,
        }
    if "apiconnectionerror" in lowered or "connection error" in lowered or "connecterror" in lowered:
        return {
            "kind": "connection",
            "message": "Provider connection failed. This is usually network, DNS, TLS, VPN, or provider availability, not billing.",
            "detail": text,
        }
    return {
        "kind": "provider_or_runtime_error",
        "message": "Provider/runtime call failed.",
        "detail": text,
    }
