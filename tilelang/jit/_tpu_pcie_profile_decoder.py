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
import json
from pathlib import Path
import sys


_SUPPORTED_ARCHES = frozenset(("tpub_7_1", "tpub_7_1_e", "tpub_7_1_e_rv"))
_DECODED_REPORT_NAME = "tilelang_pcie_profile.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--arch", choices=sorted(_SUPPORTED_ARCHES), required=True)
    return parser.parse_args()


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
                    raise ValueError(
                        f"Unexpected bigTpuProfile {engine} event: {record!r}")
                info, detail, metadata = record[:3]
                if not isinstance(info, dict) or not isinstance(metadata, dict):
                    raise ValueError(
                        f"Unexpected bigTpuProfile {engine} event fields: {record!r}")
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
    profile_dir = args.profile_dir.resolve()
    raw_files = tuple(sorted(
        path for path in profile_dir.glob("cdm_profile_data_dev*")
        if path.exists()))
    if not raw_files:
        print("No cdm_profile_data_dev* input was found.", file=sys.stderr)
        return 2

    try:
        from bigTpuProfile.bmprofile_perfAI_2260 import BMProfileParserPerfAI
    except ImportError as exc:
        print(
            "PCIe decoding requires preinstalled bigTpuProfile; "
            f"automatic installation is disabled ({exc}).",
            file=sys.stderr,
        )
        return 3
    for index, raw_path in enumerate(raw_files):
        output_dir = profile_dir / f"decoded_{index}"
        parser = BMProfileParserPerfAI()
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
            "events": events,
        }
        (output_dir / _DECODED_REPORT_NAME).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"decoded arch={args.arch} input={raw_path} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
