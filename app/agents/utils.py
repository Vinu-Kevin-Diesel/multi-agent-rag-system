"""Shared utilities for agent LLM calls."""


def extract_content(response) -> str:
    """Extract text content from an OpenAI-compatible chat completion response.

    Handles cases where content may be None (e.g., Kimi K2.5 thinking mode
    puts output in reasoning_content instead of content).
    """
    choice = response.choices[0]
    msg = choice.message

    # Standard content field
    if msg.content:
        return msg.content

    # Some models use reasoning_content for thinking models
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return msg.reasoning_content

    # Try to get from model_extra (catches non-standard fields)
    if hasattr(msg, "model_extra") and msg.model_extra:
        for key in ("reasoning_content", "thinking", "thought"):
            if key in msg.model_extra and msg.model_extra[key]:
                return msg.model_extra[key]

    return ""
