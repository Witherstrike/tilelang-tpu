# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Internal Linux process-tree supervisor for TPU profile commands.

``TPUInstructionProfiler`` runs this helper as the direct child in a new,
private session/process group.  The helper arms Linux's parent-death signal
after checking its expected parent PID, and the actual profile command stays
in that same process group.  This arrangement matters when an outer watchdog
kills pytest/the profiler: the helper receives ``SIGTERM`` from the kernel and
kills the whole worker group before it exits.

It is deliberately a small, standalone script rather than an import of the
TileLang package: the profiling worker changes cwd to its private artifact
directory, and this supervisor must remain executable even when the caller
did not export the source checkout through ``PYTHONPATH``.
"""

from __future__ import annotations

from contextlib import suppress
import os
import signal
import subprocess
import sys
from typing import Optional, Sequence

_child: Optional[subprocess.Popen] = None
_PR_SET_PDEATHSIG = 1


def _stop_child_group(signum: int, _frame: object) -> None:
    """Kill this private worker process group before the helper exits."""

    with suppress(ProcessLookupError):
        # The target deliberately inherits this supervisor's process group.
        # This also catches ordinary compiler/AutoRunner descendants, whereas
        # PR_SET_PDEATHSIG alone only reaches the direct supervisor process.
        os.killpg(os.getpgrp(), signal.SIGKILL)
    # Do not run Python cleanup handlers after a parent-death event.  In
    # particular, waiting for arbitrary CModel/PerfAI children here would
    # defeat the outer watchdog's termination guarantee.
    os._exit(128 + signum)


def _arm_parent_death_signal(expected_parent_pid: int) -> bool:
    """Return whether the original profiler is still this helper's parent.

    This uses a before/after parent-PID check around ``prctl``.  It avoids a
    ``preexec_fn`` in the profiler (unsafe in a threaded Python parent), while
    closing the setup race: if the profiler has already died, this helper
    exits before it can launch a worker.
    """

    if not sys.platform.startswith("linux"):
        print("TPU profile supervisor requires Linux PR_SET_PDEATHSIG", file=sys.stderr)
        return False
    if os.getppid() != expected_parent_pid:
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = (
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        )
        prctl.restype = ctypes.c_int
        if prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
            print("Could not set PR_SET_PDEATHSIG for TPU profile supervisor", file=sys.stderr)
            return False
    except (AttributeError, ImportError, OSError) as exc:
        print(f"Could not access Linux prctl for TPU profile supervisor: {exc}", file=sys.stderr)
        return False

    # A blocked inherited SIGTERM would otherwise delay parent-death handling.
    # If the parent died while it was blocked, unblocking before a worker exists
    # is safe; the final parent check covers an ignored/default signal race.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
    return os.getppid() == expected_parent_pid


def _main(argv: Sequence[str]) -> int:
    if len(argv) < 4 or argv[0] != "--parent-pid" or argv[2] != "--":
        print(
            "usage: _tpu_profile_supervisor.py --parent-pid <pid> -- <command> [args ...]",
            file=sys.stderr,
        )
        return 2
    try:
        expected_parent_pid = int(argv[1])
    except ValueError:
        print("TPU profile supervisor parent pid must be an integer", file=sys.stderr)
        return 2
    if expected_parent_pid <= 1 or not _arm_parent_death_signal(expected_parent_pid):
        return 125

    # The parent-death signal is SIGTERM.  Also make the normal signal paths
    # use the same tree cleanup, because TPUInstructionProfiler's own timeout
    # terminates this supervisor's process group.
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, _stop_child_group)

    # Catch a parent death that landed in the small interval between arming
    # PR_SET_PDEATHSIG and installing the Python handler.
    if os.getppid() != expected_parent_pid:
        return 125

    global _child
    try:
        # Do not start a new session here.  The outer profiler already gave us
        # a private process group, and keeping the command in it lets the
        # signal handler kill the worker and ordinary descendants together.
        _child = subprocess.Popen(argv[3:])
    except OSError as exc:
        print(f"Could not start guarded TPU profile command: {exc}", file=sys.stderr)
        return 127

    returncode = _child.wait()
    # Shell-compatible status helps the parent distinguish a target that was
    # externally signalled from a normal nonzero command result.
    return 128 + -returncode if returncode < 0 else returncode


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
