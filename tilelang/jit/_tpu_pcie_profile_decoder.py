# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Offline TPUv7 PCIe profile decoder used by ``tpu_profiling``.

This helper deliberately does not reproduce PPL's automatic ``pip install``
fallback.  Profiling a board and changing the Python environment are separate
operations; a missing decoder is reported to the caller while raw recorder
artifacts remain intact.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys

_SUPPORTED_ARCHES = frozenset(("tpub_7_1", "tpub_7_1_e", "tpub_7_1_e_rv"))
_DECODED_REPORT_NAME = "tilelang_pcie_profile.json"
_IDENTITY_PREFIX = "TILELANG_TPU_PCIE_DECODER_IDENTITY="


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--profile-dir", type=Path)
    parser.add_argument("--arch", choices=sorted(_SUPPORTED_ARCHES))
    args = parser.parse_args()
    if args.preflight and args.arch is not None:
        parser.error("--arch is not valid with --preflight")
    if args.profile_dir is not None and args.arch is None:
        parser.error("--arch is required with --profile-dir")
    return args


def _load_decoder():
    """Load and validate the one structured TPUv7 API consumed by TileLang."""

    import bigTpuProfile
    from bigTpuProfile.bmprofile_perfAI_2260 import BMProfileParserPerfAI

    parse_method = getattr(BMProfileParserPerfAI, "parse", None)
    if not callable(parse_method):
        raise AttributeError("BMProfileParserPerfAI.parse is absent or not callable")
    package_version = getattr(bigTpuProfile, "__version__", None)
    if package_version is None:
        try:
            package_version = metadata.version("bigTpuProfile")
        except metadata.PackageNotFoundError:
            package_version = "unknown"
    identity = {
        "package": "bigTpuProfile",
        "package_version": str(package_version),
        "parser_api": (
            f"{BMProfileParserPerfAI.__module__}."
            f"{BMProfileParserPerfAI.__qualname__}.parse"),
    }
    return BMProfileParserPerfAI, identity


def _print_identity(identity) -> None:
    print(_IDENTITY_PREFIX + json.dumps(identity, ensure_ascii=True, sort_keys=True))


def _canonical_events(result):
    """Normalize the current bigTpuProfile result without depending on PerfAI UI."""

    events = []
    engine_groups = (
        ("bdc", getattr(result, "bd_events", None)),
        ("gdma", getattr(result, "gdma_events", None)),
        ("sdma", getattr(result, "sdma_events", None)),
        ("cdma", getattr(result, "cdma_events", None)),
    )
    if not any(groups is not None for _, groups in engine_groups):
        return None
    for engine, groups in engine_groups:
        for group in groups or ():
            for record in group:
                if not isinstance(record, (tuple, list)) or len(record) < 3:
                    raise ValueError(f"Unexpected bigTpuProfile {engine} event: {record!r}")
                info, detail, metadata = record[:3]
                if not isinstance(info, dict) or not isinstance(metadata, dict):
                    raise ValueError(f"Unexpected bigTpuProfile {engine} event fields: {record!r}")
                begin = info.get("Start Time(ns)")
                end = info.get("End Time(ns)")
                if not isinstance(begin, (int, float)) or not isinstance(end, (int, float)):
                    raise ValueError(
                        f"bigTpuProfile {engine} event has no numeric nanosecond range")
                events.append({
                    "engine": engine,
                    "begin": begin,
                    "end": end,
                    "unit": "ns",
                    "core_id": metadata.get("Core Id"),
                    "command_id": info.get("Cmd Id"),
                    "opcode": info.get("Function Name"),
                    "fields": {
                        "info": info,
                        "detail": detail,
                        "metadata": metadata,
                    },
                })
    return events


def main() -> int:
    args = _parse_args()
    try:
        parser_class, decoder_identity = _load_decoder()
    except ImportError as exc:
        print(
            "PCIe decoding requires preinstalled bigTpuProfile; "
            f"automatic installation is disabled ({exc}).",
            file=sys.stderr,
        )
        return 3
    except (AttributeError, TypeError) as exc:
        print(
            "The installed bigTpuProfile does not provide the callable "
            f"BMProfileParserPerfAI.parse API required by TileLang ({exc}).",
            file=sys.stderr,
        )
        return 4

    if args.preflight:
        _print_identity(decoder_identity)
        return 0

    profile_dir = args.profile_dir.resolve()
    raw_files = tuple(
        sorted(path for path in profile_dir.glob("cdm_profile_data_dev*") if path.exists()))
    if not raw_files:
        print("No cdm_profile_data_dev* input was found.", file=sys.stderr)
        return 2

    for index, raw_path in enumerate(raw_files):
        output_dir = profile_dir / f"decoded_{index}"
        parser = parser_class()
        result = parser.parse(str(raw_path))
        events = _canonical_events(result)
        if events is None:
            print(
                "The installed bigTpuProfile API does not expose the structured "
                "ProfileResult required by TileLang.",
                file=sys.stderr,
            )
            return 4
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "arch": args.arch,
            "source": raw_path.name,
            "decoder_identity": decoder_identity,
            "events": events,
        }
        (output_dir / _DECODED_REPORT_NAME).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"decoded arch={args.arch} input={raw_path} output={output_dir}")
    _print_identity(decoder_identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
