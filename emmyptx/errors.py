"""The one exception every stage raises for input it cannot compile."""


class CompileError(Exception):
    """The IR uses something this compiler does not support (or is malformed)."""
