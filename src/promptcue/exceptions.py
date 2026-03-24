# promptcue | PromptCue-specific exception hierarchy
# Maintainer: Informity


class PromptCueError(Exception):
    """Base exception for all PromptCue errors."""


class PromptCueRegistryError(PromptCueError):
    """Raised when the query type registry cannot be loaded."""
