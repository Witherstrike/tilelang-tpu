"""Compare local-memory allocation recorded in PPL final MLIR.

Addresses are not generally unique, so the default comparison checks sizes,
bank spans, live ranges, simultaneous address overlap, and legal address reuse.
Use ``exact_addresses=True`` when reproducing one specific compiler result.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


_LOCAL_TENSOR = re.compile(r"ppl\.tensorbe\s+LOCAL\b.*?\{(?P<attrs>.*?)\}")
_INT_ATTR = r"\b{name}\s*=\s*(-?\d+)\s*:\s*i\d+"
_RANGE_ATTR = re.compile(r"\blive_range\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]")
_NAME_ATTR = re.compile(r'\bppl\.vname\s*=\s*"([^"]+)"')
_INDEX_ATTR = re.compile(r"\bidx\s*=\s*(\d+)\s*:\s*i\d+")
_CONFLICT_ATTR = re.compile(r"\bbank_conflict\s*=\s*\[([^]]*)\]")


@dataclass(frozen=True)
class LocalTensor:
    name: str
    address: int
    size: int
    live_start: int
    live_end: int
    bank_start: int
    bank_end: int
    conflicts: frozenset[str] = frozenset()


def _int_attr(attrs: str, name: str) -> int:
    match = re.search(_INT_ATTR.format(name=re.escape(name)), attrs)
    if not match:
        raise ValueError(f"missing {name!r} in local tensor attributes: {attrs}")
    return int(match.group(1))


def parse_final_mlir(text: str, *, bank_size: int = 16 * 1024) -> dict[str, LocalTensor]:
    """Parse allocation facts from PPL ``tensorbe LOCAL`` operations."""
    records = []
    index_to_name = {}
    for match in _LOCAL_TENSOR.finditer(text):
        attrs = match.group("attrs")
        name_match = _NAME_ATTR.search(attrs)
        range_match = _RANGE_ATTR.search(attrs)
        if not name_match or not range_match:
            raise ValueError(f"local tensor lacks name/live_range: {attrs}")
        name = name_match.group(1)
        address = _int_attr(attrs, "address")
        size = _int_attr(attrs, "size")
        if address < 0 or size <= 0:
            raise ValueError(f"invalid allocation for {name}: address={address}, size={size}")
        if name in index_to_name.values():
            raise ValueError(f"duplicate local tensor name: {name}")
        index_match = _INDEX_ATTR.search(attrs)
        if index_match:
            index_to_name[int(index_match.group(1))] = name
        conflict_match = _CONFLICT_ATTR.search(attrs)
        conflict_indices = tuple(
            int(value.strip()) for value in conflict_match.group(1).split(",")
            if value.strip()
        ) if conflict_match else ()
        records.append((name, address, size, range_match, conflict_indices))
    tensors: dict[str, LocalTensor] = {}
    for name, address, size, range_match, conflict_indices in records:
        tensors[name] = LocalTensor(
            name=name,
            address=address,
            size=size,
            live_start=int(range_match.group(1)),
            live_end=int(range_match.group(2)),
            bank_start=address // bank_size,
            bank_end=(address + size - 1) // bank_size,
            conflicts=frozenset(index_to_name[index] for index in conflict_indices),
        )
    return tensors


def parse_tilelang_attrs(
    attrs: Mapping[str, object], *, bank_size: int = 16 * 1024
) -> dict[str, LocalTensor]:
    """Parse ``AddressAssign`` allocation summary PrimFunc attributes."""
    prefix = "tir.tpu.lmem."
    fields: dict[str, dict[str, object]] = {}
    for raw_key, value in attrs.items():
        key = str(raw_key)
        if not key.startswith(prefix):
            continue
        name, field = key[len(prefix):].rsplit(".", 1)
        fields.setdefault(name.lower(), {})[field] = value
    tensors = {}
    for name, values in fields.items():
        address = int(values["address"])
        size = int(values["size"])
        conflicts = frozenset(
            item.lower() for item in str(values.get("conflicts", "")).split(",")
            if item
        )
        tensors[name] = LocalTensor(
            name=name,
            address=address,
            size=size,
            live_start=int(values["live_start"]),
            live_end=int(values["live_end"]),
            bank_start=address // bank_size,
            bank_end=(address + size - 1) // bank_size,
            conflicts=conflicts,
        )
    return tensors


def _overlap(lhs_start: int, lhs_end: int, rhs_start: int, rhs_end: int) -> bool:
    return max(lhs_start, rhs_start) < min(lhs_end, rhs_end)


def validate_allocations(
    tensors: dict[str, LocalTensor], *, bank_size: int, bank_num: int = 16,
    alignment: int = 64,
) -> None:
    """Check chip-level capacity, alignment, and simultaneous-liveness safety."""
    for tensor in tensors.values():
        assert tensor.address % alignment == 0, tensor.name
        assert tensor.address + tensor.size <= bank_size * bank_num, tensor.name
    names = sorted(tensors)
    for index, lhs_name in enumerate(names):
        lhs = tensors[lhs_name]
        for rhs_name in names[index + 1 :]:
            rhs = tensors[rhs_name]
            live_overlap = _overlap(lhs.live_start, lhs.live_end,
                                    rhs.live_start, rhs.live_end)
            address_overlap = _overlap(lhs.address, lhs.address + lhs.size,
                                       rhs.address, rhs.address + rhs.size)
            if address_overlap and live_overlap:
                raise AssertionError(
                    f"illegal simultaneous allocation overlap: {lhs_name}, {rhs_name}")
            has_conflict = (rhs_name in lhs.conflicts or
                            lhs_name in rhs.conflicts)
            bank_overlap = _overlap(lhs.bank_start, lhs.bank_end + 1,
                                    rhs.bank_start, rhs.bank_end + 1)
            if has_conflict and live_overlap and bank_overlap:
                raise AssertionError(
                    f"live conflict shares bank span: {lhs_name}, {rhs_name}")


def compare_final_mlir(
    actual: str,
    expected: str,
    *,
    bank_size: int = 16 * 1024,
    exact_addresses: bool = False,
) -> None:
    """Assert that two final MLIR allocations have the same invariants."""
    lhs = parse_final_mlir(actual, bank_size=bank_size)
    rhs = parse_final_mlir(expected, bank_size=bank_size)
    compare_allocations(lhs, rhs, bank_size=bank_size,
                        exact_addresses=exact_addresses)


def compare_allocations(
    lhs: dict[str, LocalTensor], rhs: dict[str, LocalTensor], *,
    bank_size: int = 16 * 1024, exact_addresses: bool = False,
) -> None:
    """Compare TileLang and PPL allocations without assuming one address solution."""
    validate_allocations(lhs, bank_size=bank_size)
    validate_allocations(rhs, bank_size=bank_size)
    assert lhs.keys() == rhs.keys(), (sorted(lhs), sorted(rhs))
    lhs_origin = min((item.live_start for item in lhs.values()), default=0)
    rhs_origin = min((item.live_start for item in rhs.values()), default=0)
    for name in lhs:
        actual_tensor, expected_tensor = lhs[name], rhs[name]
        assert actual_tensor.size == expected_tensor.size, name
        assert (actual_tensor.live_start - lhs_origin,
                actual_tensor.live_end - lhs_origin) == (
            expected_tensor.live_start - rhs_origin,
            expected_tensor.live_end - rhs_origin), name
        assert actual_tensor.bank_end - actual_tensor.bank_start == (
            expected_tensor.bank_end - expected_tensor.bank_start), name
        # TileLang records conflicts that constrain its bank-aware heuristic;
        # PPL may conservatively record additional producer/consumer edges.
        assert actual_tensor.conflicts <= expected_tensor.conflicts, name
        if exact_addresses:
            assert actual_tensor.address == expected_tensor.address, name


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("actual", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--bank-size", type=int, default=16 * 1024)
    parser.add_argument("--exact-addresses", action="store_true")
    args = parser.parse_args(argv)
    compare_final_mlir(
        args.actual.read_text(encoding="utf-8"),
        args.expected.read_text(encoding="utf-8"),
        bank_size=args.bank_size,
        exact_addresses=args.exact_addresses,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
