# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Shared, dependency-free metadata helpers for TPU matrix runners."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any


def git_source_identity(repo_root: Path) -> dict[str, Any]:
    """Record the tracked implementation state used to launch a matrix.

    Research prose and ignored trace artifacts do not affect generated code,
    so the dirty check deliberately excludes ``research/**``.  Missing Git is
    not silently treated as a clean checkout: the summary retains null
    identity fields plus a diagnostic so consumers can reject unverifiable
    evidence.
    """

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if not revision:
            raise RuntimeError("git returned an empty HEAD revision")
        tracked_status = subprocess.run(
            [
                "git", "status", "--porcelain=v1", "--untracked-files=no",
                "--", ".", ":(exclude)research/**",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        return {
            "git_commit": None,
            "implementation_worktree_dirty": None,
            "source_identity_scope": "tracked files excluding research/**",
            "git_identity_error": f"{type(error).__name__}: {error}",
        }
    return {
        "git_commit": revision,
        "implementation_worktree_dirty": bool(tracked_status.strip()),
        "source_identity_scope": "tracked files excluding research/**",
    }
