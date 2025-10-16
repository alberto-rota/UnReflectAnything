from contextlib import AbstractContextManager


class Ablation(AbstractContextManager):
    """
    Lightweight conditional context for ablation blocks.

    Usage inside `Engine` methods:
      - Conditional block:
            with self.ablation:
                # code that conceptually belongs to the ablation setting
      - Or directly:
            if self.ablation:
                ...

    Note: Python `with` cannot truly skip execution of the block. This context
    simply carries the enabled flag and integrates neatly with existing code.
    Prefer `if self.ablation:` when you need to guard execution.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __bool__(self):
        return self.enabled

    def set(self, enabled: bool):
        self.enabled = bool(enabled)


