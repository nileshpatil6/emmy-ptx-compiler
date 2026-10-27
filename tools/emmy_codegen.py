"""Expose Emmy's unmodified compile command without unrelated POSIX-only CLIs.

Emmy 0.3.11's top-level CLI imports the trace command, which imports fcntl
on Windows. The organisers' generator only needs compile --ir cuda. This
entry point registers that same command and does no lowering of its own.
"""

import argparse
import sys
import types


def windows_flock(handle, operation):
    """Only the exclusive file lock used by Emmy's golden writer is supported."""
    import msvcrt

    if operation != 2:
        raise NotImplementedError("Only LOCK_EX is supported by this generator adapter")
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)


if sys.platform == "win32":
    compat = types.ModuleType("fcntl")
    compat.LOCK_EX = 2
    compat.flock = windows_flock
    sys.modules["fcntl"] = compat

from emmy.commands.compile import register_compile_command  # noqa: E402
from emmy.logging_setup import setup_cli_logging  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="ascii", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="ascii", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    register_compile_command(parser.add_subparsers(dest="command", required=True))
    setup_cli_logging()
    args = parser.parse_args()
    args.func(args)
