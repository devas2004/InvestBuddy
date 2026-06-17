"""
Swappable LLM generation wrapper.

This is the ONLY file that imports a provider SDK.  To switch providers,
re-implement generate() here — all other modules call this function and are
unaware of the provider.

Default: Google Gemini 2.5 Flash via the google-genai package.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set — check your .env file")
        _client = genai.Client(api_key=api_key)
    return _client


def generate(system: str, user: str, max_tokens: int = 1024) -> str:
    """
    Generate a text response from the configured LLM.

    Args:
        system:     System / instruction prompt.
        user:       User message (question + retrieved context).
        max_tokens: Maximum output tokens.

    Returns:
        The model's plain-text response.
    """
    client = _get_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text
