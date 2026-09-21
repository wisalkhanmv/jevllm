"""JevLLM — a language model built out of Jev, a model designed never to be one."""

__version__ = "0.2.0"

from .api import JevClient, JevError, api_key, sample  # noqa: F401
from .decode import Settings, Stats, Step, decode, text_of  # noqa: F401
