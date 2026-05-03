"""Model routing for MAGMA v2 agents."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from strands.models.anthropic import AnthropicModel
from strands.models.openai import OpenAIModel
from strands.models.gemini import GeminiModel


DEFAULT_AGENT_MODEL_ID = "gpt-5.4"
DEFAULT_AGENT_PROVIDER = "openai"


def agent_model_id(agent_name: str = "agent") -> str:
    """Return the configured model id for an agent."""
    load_dotenv()
    specific_key = f"MAGMA_{agent_name.upper()}_MODEL"
    return (
        os.getenv(specific_key)
        or os.getenv("MAGMA_AGENT_MODEL")
        or DEFAULT_AGENT_MODEL_ID
    )


def agent_provider(agent_name: str = "agent") -> str:
    """Return the configured provider for an agent: openai | anthropic | google."""
    load_dotenv()
    specific_key = f"MAGMA_{agent_name.upper()}_PROVIDER"
    return (
        os.getenv(specific_key)
        or os.getenv("MAGMA_AGENT_PROVIDER")
        or DEFAULT_AGENT_PROVIDER
    ).lower()


def create_agent_model(agent_name: str = "agent"):
    """Create a model instance for an agent.

    Provider is resolved from MAGMA_{AGENT}_PROVIDER or MAGMA_AGENT_PROVIDER.
    Supported values: openai (default), anthropic, google.
    """
    load_dotenv()
    model_id = agent_model_id(agent_name)
    provider = agent_provider(agent_name)

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        client_args = {"api_key": api_key} if api_key else {}
        return AnthropicModel(model_id=model_id, client_args=client_args)

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        client_args = {"api_key": api_key} if api_key else {}
        return GeminiModel(model_id=model_id, client_args=client_args)

    api_key = os.getenv("OPENAI_API_KEY")
    client_args = {"api_key": api_key} if api_key else {}
    return OpenAIModel(model_id=model_id, client_args=client_args)
