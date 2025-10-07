"""Envector LangChain integration package.

Provides a LangChain-compatible VectorStore that wraps the high-level `es2` SDK.
All code and comments are in English as per project rules.
"""

from .vectorstore import Envector
from .config import ConnectionConfig, EnvectorConfig, IndexSettings, KeyConfig

__all__ = ["Envector", "ConnectionConfig", "EnvectorConfig", "IndexSettings", "KeyConfig"]

